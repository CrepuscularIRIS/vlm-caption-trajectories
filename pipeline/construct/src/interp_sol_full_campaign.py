"""Quota-window continuation of NEVER-ISSUED episodes; immutable stopped runs.

Uses the frozen constructor unchanged. No semantic review, no retry, no model
probe. Each model/task gets at most one episode across this campaign.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import interp_sol_full_4040 as b
from tiered_transport import Ledger, ROOT, write_json

PARENT = b.DEFAULT_PARENT
REGISTRY = 'CONTINUATION_PLAN.json'


def locations(parent):
    plan = b.read(parent/REGISTRY)
    return [parent] + [Path(w['parent']) for w in plan['waves']]


def inventory(parent):
    issued = {r:set() for r in b.ROLES}
    results = {r:{} for r in b.ROLES}
    for loc in locations(parent):
        b.verify(loc)
        for role in b.ROLES:
            out = loc/role
            if not out.exists(): continue
            if b.pending_requests(loc,role):
                raise ValueError('Unsettled requests: '+str(out))
            local={d['label'].split('/')[0] for d in b.old.read_rows(out/'budget/dispatch.jsonl')} if (out/'budget/dispatch.jsonl').exists() else set()
            if issued[role] & local: raise ValueError('Duplicate model/task episode')
            issued[role].update(local)
            confirmed = set(b.choose_ids(loc,b.ROLES[b.ROLES.index(role)+1])) if role!='gemini' else set()
            for path in (out/role/'results').glob('*.json'):
                result=b.read(path)
                results[role][result['audit_id']] = dict(result=result,path=str(path),parent=str(loc),
                                                        fallback_confirmed=result['audit_id'] in confirmed)
    return issued,results


def eligible_ids(parent,role):
    issued,results=inventory(parent)
    ids=[s['audit_id'] for s in b.read(parent/'SCOPE_ALL_4040.json')]
    if role=='sol':return [aid for aid in ids if aid not in issued[role]]
    previous=b.ROLES[b.ROLES.index(role)-1]
    return [aid for aid in ids if aid not in issued[role] and aid in results[previous]
            and results[previous][aid]['fallback_confirmed']]


def prepare_wave(parent,role):
    plan=b.read(parent/REGISTRY)
    if plan.get('blocked',{}).get(role):raise ValueError('Role requires user decision')
    ids=eligible_ids(parent,role)
    if not ids:return None
    for w in plan['waves']:
        if w['role']==role:
            loc=Path(w['parent']);state=loc/role/'RUN_STATE.json'
            if not state.exists() or b.read(state)['state']=='running':return loc
    n=1+sum(w['role']==role for w in plan['waves'])
    loc=parent/'continuations'/f'{role}_{n:03d}'
    if loc.exists():raise ValueError('Unregistered continuation directory')
    loc.mkdir(parents=True)
    allscope=b.read(parent/'SCOPE_ALL_4040.json');private=b.read(parent/'SELECTION_PRIVATE.json');idset=set(ids)
    scope=[s for s in allscope if s['audit_id'] in idset]
    write_json(loc/'SCOPE_ALL_4040.json',scope)
    write_json(loc/'SELECTION_PRIVATE.json',[p for p in private if p['audit_id'] in idset])
    write_json(loc/'SOURCE_PIXEL_HASHES.json',b.read(parent/'SOURCE_PIXEL_HASHES.json'))
    hashes={};original_hashes=b.read(parent/'INPUT_HASHES.json')
    for aid in ids:
        # Preserve preflight pixel hashes. Verify native/prefix bytes again at
        # actual dispatch, without reading all native images during scheduling.
        source=parent/'inputs'/(aid+'.json')
        if b.old.sha(source)!=original_hashes[str(source)]:raise ValueError('Input changed')
        packet=b.read(source);path=loc/'inputs'/(aid+'.json')
        write_json(path,packet);hashes[str(path)]=b.old.sha(path)
    write_json(loc/'INPUT_HASHES.json',hashes)
    cfg=b.read(parent/'config.json')
    cfg.update(tasks=len(ids),continuation_of=str(parent),active_role=role,
               task_policy='Only never-issued model/task episodes across all campaign locations',
               not_before_utc=plan['not_before_utc'][role])
    cfg['frozen_files']=dict(cfg['frozen_files'])
    for path in [Path(__file__).resolve(),parent/'config.json',parent/'CONFIG.sha256',
                 loc/'SCOPE_ALL_4040.json',loc/'SELECTION_PRIVATE.json',loc/'SOURCE_PIXEL_HASHES.json',loc/'INPUT_HASHES.json']:
        cfg['frozen_files'][str(path)]=b.old.sha(path)
    cfg['models']={role:dict(cfg['models'][role],call_cap=10*len(ids))}
    write_json(loc/'config.json',cfg);(loc/'CONFIG.sha256').write_text(b.old.sha(loc/'config.json')+'\n')
    out=loc/role;out.mkdir()
    branch=dict(cfg,frozen_files=dict(cfg['frozen_files'],**{str(loc/'config.json'):b.old.sha(loc/'config.json')}))
    write_json(out/'config.json',branch);(out/'CONFIG.sha256').write_text(b.old.sha(out/'config.json')+'\n')
    write_json(loc/'SELECTION_STATS.json',dict(tasks=len(ids),role=role,ids=ids,source_preflight_reused=True))
    plan['waves'].append(dict(role=role,parent=str(loc),tasks=len(ids),created_at=b.now(),
                              not_before_utc=cfg['not_before_utc']))
    write_json(parent/REGISTRY,plan)
    return loc


def launch_wave(parent,loc,role):
    cfg=b.verify(loc,role);out=loc/role
    if time.time()<datetime.fromisoformat(cfg['not_before_utc']).timestamp():raise ValueError('Quota window not elapsed; zero calls')
    if time.time()>cfg['deadline']:raise ValueError('Campaign deadline elapsed')
    if (parent/'PAUSE').exists() or (loc/'PAUSE').exists():raise ValueError('Campaign paused')
    if any((out/n).exists() for n in ('LAUNCH.json','RUN_STATE.json','PAUSE')):raise ValueError('Never relaunch an issued wave')
    plan=b.read(parent/REGISTRY)
    if plan.get('blocked',{}).get(role):raise ValueError('Role blocked')
    if role=='gemini':b.start_gemini_proxy(loc)
    cmd=['${PYTHON}','-u',str(Path(__file__).resolve()),'run',
         '--parent',str(parent),'--wave',str(loc),'--role',role]
    with (out/'worker.log').open('ab') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'LAUNCH.json',dict(pid=proc.pid,command=cmd,at=b.now(),first_real_receipt_pending=True))
    return dict(launched=True,role=role,tasks=cfg['tasks'],first_real_receipt_pending=True)


def run_wave(parent,loc,role):
    cfg=b.verify(loc,role);out=loc/role
    if time.time()<datetime.fromisoformat(cfg['not_before_utc']).timestamp():raise ValueError('Quota gate')
    with (out/'RUN.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'RUN_STATE.json').exists():raise ValueError('Fresh wave only')
        issued,_=inventory(parent)
        ids=[s['audit_id'] for s in b.read(loc/'SCOPE_ALL_4040.json')]
        if issued[role]&set(ids):raise ValueError('Previously issued tasks in wave')
        private={p['audit_id']:p for p in b.read(loc/'SELECTION_PRIVATE.json')}
        ledger=Ledger(out/'budget',{s['model']:s['call_cap'] for s in cfg['models'].values()})
        api=b.Transport(out,role,cfg,ledger);queue=iter(ids);lock=threading.Lock();started=set()
        write_json(out/'RUN_STATE.json',dict(state='running',started_at=b.now(),pid=os.getpid()))
        def next_id():
            with lock:
                aid=next(queue,None)
                if aid:started.add(aid)
                return aid
        def one(aid):
            try:
                if (parent/'PAUSE').exists():raise b.BudgetStop('Campaign PAUSE')
                b.verify(loc,role);api.check()
                result=b.generate(loc,b.packet(loc,aid),api,private[aid])
                print(json.dumps(dict(audit_id=aid,model=api.model,termination=result['rollout']['termination'],
                                 interpolation=result['interpolation_inserted'])),flush=True)
                return result
            except Exception as ex:
                write_json(out/'task_errors'/(aid+'.json'),dict(audit_id=aid,error_type=type(ex).__name__,at=b.now()))
                keys=[d['request_sha256'] for d in ledger.rows() if d['label'].split('/')[0]==aid]
                if keys:
                    cp=out/role/'checkpoints'/(aid+'.json')
                    ro=b.read(cp)['rollout'] if cp.exists() else dict(steps=[],views=[],crops=0,answer=None,export=None)
                    ro.update(termination='episode_interrupted',export=None,crops=max(0,len(ro.get('views',[]))-1))
                    result=dict(audit_id=aid,item_id=private[aid]['manifest']['item_id'],model=api.model,assigned_role=role,
                                rollout=ro,bootstrap={},interpolation_inserted=any(s.get('origin')=='editor_forced' for s in ro['steps']),
                                episode_request_sha256s=keys,prefix_origin='incomplete_technical_episode',paired_split=private[aid]['split'],
                                semantic_review='NOT_REQUESTED_BY_USER',process_quality_unverified=True,
                                cpu=dict(passed=False,reasons=[type(ex).__name__]),
                                objective_checks=dict(eligible_without_process_review=False,protocol_passed=False,GT_score={'score':None},
                                                     editor_supervision_flags_correct=False,actual_token_mask_verified=False,target_4B_loader_verified=False),
                                needs_fallback=True,fallback_reason='episode_interrupted:'+type(ex).__name__)
                    write_json(out/role/'results'/(aid+'.json'),result)
                ledger.stop(api.model,'Episode exception; no same-role retry')
                return None
        first=next_id();result=one(first) if first else None
        first_ok=result is not None and result['rollout']['termination'] not in ('api_error','editor_transport_error')
        if first and not first_ok:ledger.stop(api.model,'First real transport/input failure')
        if first_ok:
            receipts=[b.read(p) for p in (out/role/'calls').glob('*.json')]
            write_json(out/'FIRST_RECEIPT_CHECK.json',dict(at=b.now(),task=first,actual_receipt_models=[c.get('returned_model') for c in receipts],
                        images=[c['images'] for c in receipts],process_quality_verified=False))
        def worker():
            while first_ok:
                try:
                    if (parent/'PAUSE').exists():return
                    api.check()
                except b.BudgetStop:return
                aid=next_id()
                if not aid:return
                one(aid)
        if first_ok:
            with ThreadPoolExecutor(max_workers=cfg['models'][role]['workers']) as pool:
                for future in [pool.submit(worker) for _ in range(cfg['models'][role]['workers'])]:future.result()
        stopped=(out/'budget'/('STOP_'+api.model)).exists() or (out/'PAUSE').exists() or (loc/'PAUSE').exists() or (parent/'PAUSE').exists()
        write_json(out/'RUN_STATE.json',dict(state='stopped_incomplete' if stopped else 'generation_complete_unreviewed',
                    finished_at=b.now(),started_tasks=len(started),saved_results=len(list((out/role/'results').glob('*.json'))),pending=b.pending_requests(loc,role)))


def tick(parent,launch=False):
    b.verify(parent);plan=b.read(parent/REGISTRY);events=[]
    if (parent/'PAUSE').exists():return dict(state='paused',events=[])
    # Close the launch-to-RUN_STATE race and do not snapshot another branch's
    # requests while they are in flight. Concurrency is within each model wave;
    # model phases are serialized, with settled upstream snapshots.
    active=[]
    for loc in locations(parent):
        for role in b.ROLES:
            state=loc/role/'RUN_STATE.json';launched=loc/role/'LAUNCH.json'
            if state.exists() and b.read(state)['state']=='running':
                active.append(dict(role=role,parent=str(loc),state='running'))
            elif launched.exists() and not state.exists():
                active.append(dict(role=role,parent=str(loc),state='launch_unconfirmed'))
    if active:return dict(at=b.now(),state='waiting_active_wave_settlement',events=active)
    for role in b.ROLES:
        plan=b.read(parent/REGISTRY)
        running=[]
        for loc in locations(parent):
            state=loc/role/'RUN_STATE.json'
            if state.exists() and b.read(state)['state']=='running':running.append(str(loc))
        if running:events.append(dict(role=role,state='running',locations=running));continue
        if plan.get('blocked',{}).get(role):events.append(dict(role=role,state='blocked',reason=plan['blocked'][role]));continue
        # A stopped post-window wave does not trigger successive quota probes.
        stopped=[w for w in plan['waves'] if w['role']==role and (Path(w['parent'])/role/'RUN_STATE.json').exists()
                 and b.read(Path(w['parent'])/role/'RUN_STATE.json')['state']=='stopped_incomplete']
        if stopped:
            plan['blocked'][role]='Post-window wave stopped; inspect receipts before further dispatch'
            write_json(parent/REGISTRY,plan);events.append(dict(role=role,state='blocked',reason=plan['blocked'][role]));continue
        if time.time()>b.read(parent/'config.json')['deadline']:events.append(dict(role=role,state='deadline_elapsed'));continue
        if time.time()<datetime.fromisoformat(plan['not_before_utc'][role]).timestamp():
            events.append(dict(role=role,state='waiting_window',not_before=plan['not_before_utc'][role]));continue
        try:
            loc=prepare_wave(parent,role)
            if not loc:events.append(dict(role=role,state='no_confirmed_work'));continue
            if launch:
                events.append(launch_wave(parent,loc,role))
                return dict(at=b.now(),events=events)
            else:events.append(dict(role=role,state='ready',wave=str(loc)))
        except (ValueError,BlockingIOError) as ex:
            events.append(dict(role=role,state='requires_inspection',reason=str(ex)))
    return dict(at=b.now(),events=events)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare-wave','tick','run']);ap.add_argument('--parent',type=Path,default=PARENT)
    ap.add_argument('--role',choices=b.ROLES);ap.add_argument('--wave',type=Path);ap.add_argument('--launch',action='store_true');a=ap.parse_args()
    if a.action=='prepare-wave':value=prepare_wave(a.parent.resolve(),a.role);print(str(value))
    elif a.action=='tick':print(json.dumps(tick(a.parent.resolve(),a.launch),ensure_ascii=False))
    else:run_wave(a.parent.resolve(),a.wave.resolve(),a.role)
