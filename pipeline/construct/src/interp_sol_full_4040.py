"""Fresh, user-authorized Sol -> K3 -> Gemini 4040-task construction funnel.

No old run is resumed. No semantic judge. A forced intervention is context only;
GT is read by the offline scorer and never enters a model request.
"""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time

from dotenv import dotenv_values
from PIL import Image
import requests

import caption_v51 as proto
from corpus_scoring import score
from export_protocol_check import check_turn
import interp_recovery_calibration as old
import interp_recovery_full as full
import interp_recovery_full_r2 as editor
from probe_api import digest, structured, text_msg
from tiered_transport import BudgetStop, Ledger, ROOT, write_json

VERSION = 'sol-full-4040-autonomous-one-intervention-20261011'
DEFAULT_PARENT = ROOT/'obs/runs/interp_sol_full_4040_20261011'
ROLES = ('sol', 'k3', 'gemini')
MODEL_SPECS = {
    'sol': dict(model='gpt-5.6-sol', endpoint='http://127.0.0.1:8317', workers=32, rpm=120),
    'k3': dict(model='k3-256k', endpoint='http://127.0.0.1:4001', workers=8, rpm=30),
    'gemini': dict(model='gemini-3.8-flash', endpoint='http://127.0.0.1:4003', workers=8, rpm=30),
}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize_protocol_wire(reply, model):
    """Only an entire protocol turn in K3's wrong field; no reasoning extraction."""
    if model != 'k3-256k' or reply.get('content') or reply.get('finish_reason') != 'stop':
        return reply
    t = reply.get('reasoning_content') or ''
    if not re.fullmatch(r'\s*<(caption|think)>[^<>]*</\1>\s*<(answer|grounding)>[^<>]*</\2>\s*', t):
        return reply
    p = proto.parse_step(t)
    if p.get('step') not in range(1, 10) or p['kind'] not in ('answer', 'grounding') or check_turn(p['text']):
        return reply
    return dict(reply, content=t, original_wire_content=reply.get('content', ''),
                wire_normalization=dict(source_field='reasoning_content', source_sha256=digest(t),
                                        changed_text=False, freeform_reasoning_used=False))


def redact_error(value, secrets=()):
    t = str(value)
    for secret in secrets:
        if secret: t = t.replace(secret, '[REDACTED]')
    t = re.sub(r'data:[^;\s]+;base64,[A-Za-z0-9+/=]+', '[IMAGE_REDACTED]', t)
    t = re.sub(r'(?i)(bearer\s+)[^\s"\',;]+', r'\1[REDACTED]', t)
    t = re.sub(r'\b(?:sk|sess|key)-[A-Za-z0-9_-]+', '[REDACTED]', t)
    t = re.sub(r'(?i)((?:api[_-]?key|access_token|refresh_token|authorization|password)["\']?\s*[:=]\s*["\']?)[^\s"\',}]+', r'\1[REDACTED]', t)
    t = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[EMAIL_REDACTED]', t)
    return t[:1200]


class Transport(old.RecoveryTransport):
    def __init__(self, out, role, cfg, ledger):
        tm = old.transport.base.routing.tm
        if role == 'gemini':
            tm.MODELS[MODEL_SPECS[role]['model']] = tm.Endpoint(MODEL_SPECS[role]['endpoint'], tm.REPO_ENV, 'LITELLM_API_KEY')
        super().__init__(out, role, cfg, ledger)
        self.expected_images=read(out.parent/'SOURCE_PIXEL_HASHES.json')

    def image(self, path):
        # Returned pixels are already persisted. Do not duplicate them in media/.
        data = Path(path).read_bytes()
        image_hash=hashlib.sha256(data).hexdigest()
        with self.lock:
            expected=self.expected_images.setdefault(str(path),image_hash)
        if expected!=image_hash:
            write_json(self.batch.parent/'PAUSE',dict(reason='image_changed_before_API_delivery',path=str(path),at=now()))
            raise BudgetStop('Image bytes changed')
        with Image.open(path) as im: mime = Image.MIME[im.format]
        return dict(type='image_url', image_url=dict(url='data:'+mime+';base64,'+base64.b64encode(data).decode()))

    def check(self):
        if (self.batch.parent/'PAUSE').exists(): raise BudgetStop('Parent PAUSE')
        super().check()
        if shutil.disk_usage(self.batch).free < 5*1024**3:
            write_json(self.batch/'PAUSE', dict(reason='disk_free_below_5GiB', at=now()))
            raise BudgetStop('Disk capacity')

    def api_call(self, msgs, max_tokens, temperature, top_p):
        ep = self.ep
        payload = dict(model=self.model, messages=msgs, reasoning_effort='low',
                       temperature=ep.fixed_temperature if ep.fixed_temperature is not None else temperature,
                       top_p=ep.fixed_top_p if ep.fixed_top_p is not None else top_p,
                       max_tokens=max_tokens or self.max_tokens)
        resp = requests.post(ep.base+'/v1/chat/completions', json=payload,
                             headers={'Authorization':'Bearer '+self.key}, timeout=(20, 300))
        if resp.status_code != 200:
            try:
                body = resp.json(); err = body.get('error', body) if isinstance(body, dict) else body
                detail = {k:redact_error(err.get(k), [self.key]) for k in ('type', 'code', 'message') if err.get(k) is not None} if isinstance(err, dict) else {'message':redact_error(err, [self.key])}
            except ValueError: detail = {'message':redact_error(resp.text, [self.key])}
            e = dict(status=resp.status_code, detail=detail, response_bytes=len(resp.content),
                     response_sha256=hashlib.sha256(resp.content).hexdigest())
            quota = bool(re.search(r'usage.?limit|quota|rate.?limit|unauth|invalid.?grant|exhaust', json.dumps(detail), re.I))
            if resp.status_code in (401, 402, 403, 429) or quota:
                self.ledger.stop(self.model, 'quota/auth/rate error HTTP '+str(resp.status_code))
            # A 400 indicates an unclassified request/backend fault. Stop rather
            # than send thousands of similarly invalid contexts.
            if resp.status_code in (400, 404, 413, 422):
                self.ledger.stop(self.model, 'request/backend error HTTP '+str(resp.status_code))
            return dict(content='', reasoning_content='', finish_reason='error', returned_model=None, usage={}, errors=[e])
        data = resp.json(); choice = data['choices'][0]; m = choice['message']
        reply = dict(content=m.get('content') or '', reasoning_content=m.get('reasoning_content') or '',
                     finish_reason=choice.get('finish_reason'), returned_model=data.get('model'),
                     usage=data.get('usage', {}), response_id=data.get('id'))
        return normalize_protocol_wire(reply, self.model)


def verify(parent, role=None):
    cfg = read(parent/'config.json')
    if old.sha(parent/'config.json') != (parent/'CONFIG.sha256').read_text().strip():
        raise BudgetStop('Parent config changed')
    for path, h in cfg['frozen_files'].items():
        if old.sha(path) != h:
            write_json(parent/'PAUSE', dict(reason='frozen_source_changed', path=path, at=now()))
            raise BudgetStop('Source changed')
    if role:
        branch = read(parent/role/'config.json')
        if old.sha(parent/role/'config.json') != (parent/role/'CONFIG.sha256').read_text().strip():
            raise BudgetStop('Branch config changed')
        return branch
    return cfg


def prepare(parent):
    if parent.exists(): raise ValueError('Fresh run required')
    baseline = old.read_rows(old.BASELINE)
    if len(baseline) != 4040 or len({r['canonical_sample_key'] for r in baseline}) != 4040:
        raise ValueError('Baseline uniqueness changed')
    overlay = read(full.OVERLAY)['groups']
    prior_parent=ROOT/'obs/runs/interp_dataset2_accelerated_20261011'
    # A new full re-generation is explicitly authorized; still never race or
    # repeat requests whose predecessor status is unknown.
    for role in ('sol','k3'):
        with (prior_parent/role/'RUN.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if read(prior_parent/role/'RUN_STATE.json')['state']=='running' or pending_requests(prior_parent,role):
                raise ValueError('Previous branch not fully settled')
    parent.mkdir(parents=True)
    packets = []; private = []; preflight = {}; pixels = {}
    ordered = sorted(baseline, key=lambda r:(overlay[r['canonical_groups'][0]] != 'train', digest([VERSION, r['canonical_sample_key']])))
    for i, row in enumerate(ordered, 1):
        if old.sha(row['result_path']) != row['result_sha256']: raise ValueError('Old result changed')
        src = read(row['result_path']); image = src['item']['image']
        if image not in pixels: pixels[image] = old.sha(image)
        initial = copy.deepcopy(src['rollout']['views'][0])
        for v in src['rollout']['views'][:2]:
            if v['path'] not in pixels: pixels[v['path']] = old.sha(v['path'])
            if pixels[v['path']] != v['sha256']: raise ValueError('Old pixels changed')
        aid = f'S{i:04d}'
        p = dict(audit_id=aid, item_id=row['item_id'], question=src['item']['question'], image=image,
                 source_image_sha256=pixels[image], original_model=row['model'], original_crops=row['crops'],
                 initial_view=initial, budget=8)
        # Tool paths can remain constructible even when strong fresh Sol commits
        # immediately: only the old actually executed first crop is reusable.
        if row['crops']:
            p['fallback_visible_prefix'] = full.make_prefix(src, row, aid, 'sol')
        path = parent/'inputs'/(aid+'.json'); write_json(path, p); preflight[str(path)] = old.sha(path); packets.append(p)
        private.append(dict(audit_id=aid, manifest=row, answer_spec=src['item']['answer_spec'],
                            split=overlay[row['canonical_groups'][0]]))
    write_json(parent/'INPUT_HASHES.json', preflight)
    write_json(parent/'SOURCE_PIXEL_HASHES.json', pixels)
    write_json(parent/'SELECTION_PRIVATE.json', private)
    write_json(parent/'SCOPE_ALL_4040.json', [dict(audit_id=p['audit_id'], baseline_id=r['manifest']['audit_id'],
               canonical_sample_key=r['manifest']['canonical_sample_key'], groups=r['manifest']['canonical_groups'],
               split=r['split'], original_crops=p['original_crops']) for p, r in zip(packets, private)])
    files = [Path(__file__).resolve(), ROOT/'obs/src/build_interp_sol_full_4040.py',
             ROOT/'obs/src/interp_recovery_full_r2.py', ROOT/'obs/src/interp_recovery_full.py',
             ROOT/'obs/src/interp_recovery_calibration.py', ROOT/'obs/src/interp_recovery_v2.py',
             ROOT/'obs/src/sft_v2_topup.py', ROOT/'obs/src/caption_v51.py', ROOT/'obs/src/caption_v3.py',
             ROOT/'obs/src/export_protocol_check.py', ROOT/'obs/src/probe_replica_api.py',
             ROOT/'obs/src/probe_api.py', ROOT/'obs/src/tiered_transport.py', ROOT/'obs/src/corpus_scoring.py',
             ROOT/'obs/src/monitor_step.py', ROOT/'obs/src/run_minio3_replica.py', ROOT/'obs/src/run_caption_v51.py',
             ROOT/'obs/src/models_v51.py', ROOT/'obs/src/trajgen_models.py', old.BASELINE, full.OVERLAY,
             ROOT/'reports/interp_dataset2_accelerated_20261011/unreviewed_paired_catalog/MANIFEST.json',
             ROOT/'reports/interp_dataset2_accelerated_20261011/unreviewed_paired_catalog/TRAIN_SPLIT_SAFE_PAIRED_SHAREGPT.json',
             ROOT/'reports/interp_dataset2_accelerated_20261011/FINAL_STATS.json',
             parent/'INPUT_HASHES.json', parent/'SELECTION_PRIVATE.json', parent/'SCOPE_ALL_4040.json', parent/'SOURCE_PIXEL_HASHES.json']
    frozen = {str(f):old.sha(f) for f in files}
    cfg = dict(version=VERSION, authorization='2026-10-11 user: all possible 4040 trajectories; regenerate with Sol 5.6 first, fallback K3/Gemini; high concurrency',
               created_at=now(), deadline=time.time()+7*86400, tasks=4040, baseline_T_Q_G=[4040,4040,3690],
               budget_crops=8, max_HTTP_per_task_per_role=10, frozen_files=frozen,
               process_review='NOT_REQUESTED_BY_USER', semantic_verified=False, no_GPU=True, no_training=True,
               no_download=True, old_runs_immutable=True, target_4B_loader_verified=False,
               gemini_not_before_utc='2026-10-13T00:00:00+00:00',
               models={r:dict(s, effort='low', route='api', call_cap=40400, max_tokens=8192,
                              allowed_returned_models=[s['model']]) for r,s in MODEL_SPECS.items()},
               total_client_HTTP_cap=121200,
               retry_policy='one episode per role/task; no same-role valid/format/technical retry; unknown requests block fallback',
               fallback='confirmed objective failure OR no inserted intervention on an originally-tool task; no GT/error details supplied to next generator',
               direct_policy='fresh direct on originally-direct task is allowed; no mandatory crop/length; not classified as interpolation',
               tool_prefix_policy='fresh crop preferred; fresh direct on old tool task invokes only old first received prefix, recorded separately',
               intervention='once, after first real crop: currently visible critical patch -> adjacent real omission; no future answer/frame',
               masks='forced editor body and EOS loss0; no hidden reasoning training',
               heldout_policy='817 dev/test variations research only; original group split and evaluation remain frozen')
    write_json(parent/'config.json', cfg); (parent/'CONFIG.sha256').write_text(old.sha(parent/'config.json')+'\n')
    for role, spec in cfg['models'].items():
        branch = parent/role; branch.mkdir()
        bc = dict(cfg, models={role:spec}, examples=[], frozen_files=dict(frozen, **{str(parent/'config.json'):old.sha(parent/'config.json')}))
        write_json(branch/'config.json', bc); (branch/'CONFIG.sha256').write_text(old.sha(branch/'config.json')+'\n')
    write_json(parent/'SELECTION_STATS.json', dict(tasks=4040, groups=len({g for r in baseline for g in r['canonical_groups']}),
               direct=sum(r['crops']==0 for r in baseline), tool=sum(r['crops']>0 for r in baseline),
               split=dict(Counter(r['split'] for r in private)), original_models=dict(Counter(r['model'] for r in baseline)),
               source_pixel_files=len(pixels), preflight_verified=True))
    return cfg


def packet(parent, aid):
    p = read(parent/'inputs'/(aid+'.json'))
    expected = read(parent/'INPUT_HASHES.json')[str(parent/'inputs'/(aid+'.json'))]
    if old.sha(parent/'inputs'/(aid+'.json')) != expected: raise BudgetStop('Input changed')
    if old.sha(p['image']) != p['source_image_sha256']: raise BudgetStop('Native image changed')
    views = [p['initial_view']] + p.get('fallback_visible_prefix', {}).get('prefix_views', [])
    for v in views:
        if old.sha(v['path']) != v['sha256']: raise BudgetStop('Received image changed')
    return p


def step(ro, api, p, n, label):
    reply = api.call(p['audit_id']+'/'+label, old.as_messages(ro, api, p['question']),
                     replicate_id=VERSION+'#'+api.role+'#'+p['audit_id']+'#'+label, temperature=.7, top_p=.9)
    st = dict(step=n, request_sha256=reply['request_sha256'], finish_reason=reply['finish_reason'],
              raw_text=reply.get('content') or '', crops_before=len(ro['views'])-1,
              available_before=[v['view_id'] for v in ro['views']], origin='model_continuation', usage=reply.get('usage'))
    ro['steps'].append(st)
    if reply['finish_reason'] != 'stop':
        ro['termination'] = 'api_error' if reply['finish_reason']=='error' else 'provider_finish_error'; return None
    parsed = proto.parse_step(st['raw_text'])
    violations = proto.check_step(parsed, n, st['crops_before'], st['available_before'], 8)+check_turn(parsed['text'])
    st.update(parsed=parsed, violations=violations, flags=proto.soft_flags(parsed))
    if violations: ro['termination']='protocol_violation'; return None
    if parsed['kind']=='answer':
        ro.update(termination='abstain' if parsed['answer'].upper()=='UNCLEAR' else 'answer', answer=parsed['answer'])
    return parsed


def generate(parent, p, api, private):
    out = parent/api.role; odir = out/api.role/'obs'/p['audit_id']; odir.mkdir(parents=True, exist_ok=False)
    ro = dict(protocol=proto.PROTOCOL_VERSION, budget=8, views=[copy.deepcopy(p['initial_view'])],
              steps=[], termination='not_finished', answer=None, export=None)
    parsed = step(ro, api, p, 1, 'fresh_s1')
    bootstrap = copy.deepcopy(ro)
    prefix_origin = 'fresh_'+api.model
    inserted = False; proposal = None; proposal_reason = None
    with Image.open(p['image']) as im: native = im.convert('RGB')
    if parsed and parsed['kind']=='answer' and p['original_crops']>0 and ro['termination']=='answer':
        oldp = p['fallback_visible_prefix']
        ro = dict(protocol=proto.PROTOCOL_VERSION, budget=8, views=copy.deepcopy(oldp['prefix_views']),
                  steps=[dict(step=1, parsed=proto.parse_step(oldp['prefix_assistant']), crops_before=0,
                      available_before=['original_image'], origin='original_prefix', observation='observation_1',
                      violations=[], flags=proto.soft_flags(proto.parse_step(oldp['prefix_assistant'])))],
                  termination='not_finished', answer=None, export=None)
        prefix_origin = 'old_first_received_prefix_after_fresh_direct'
    elif parsed and parsed['kind']=='grounding':
        v, meta = old.execute_crop(native, ro['views'], parsed, 1, odir)
        ro['steps'][-1].update(meta); ro['views'].append(v)
    if ro['termination']=='not_finished' and len(ro['views'])==2:
        pp = dict(p, prefix_views=copy.deepcopy(ro['views']), prefix_assistant=ro['steps'][0]['parsed']['text'], assigned_role=api.role)
        er = api.call(
            p['audit_id']+'/editor', [text_msg('system', editor.EDITOR_SYSTEM),
                dict(role='user', content=[dict(type='text', text='Question:\n'+p['question']+'\nCompleted Step 1:\n'+pp['prefix_assistant'])]+
                     [b for v in pp['prefix_views'] for b in (dict(type='text',text='['+v['view_id']+']'),api.image(v['path']))])],
            replicate_id=VERSION+'#'+api.role+'#'+p['audit_id']+'#editor', temperature=.3, top_p=.9)
        proposal = dict(request_sha256=er['request_sha256'], raw_text=er.get('content') or '', finish_reason=er['finish_reason'])
        if er['finish_reason']!='stop': ro['termination']='editor_transport_error' if er['finish_reason']=='error' else 'editor_provider_finish_error'
        else:
            try: derived = editor.validate_proposal(structured(er.get('content') or ''), pp)
            except (ValueError, TypeError) as ex:
                derived = None; proposal_reason = str(ex); proposal['rejection_reason']=proposal_reason
            if derived:
                forced = full.pilot.forced_turn(derived)
                v, meta = old.execute_crop(native, ro['views'], forced, 2, odir)
                ro['steps'].append(dict(meta, parsed=forced, crops_before=1,
                    available_before=['original_image','observation_1'], origin='editor_forced',
                    violations=[], flags=[], request_sha256=None))
                ro['views'].append(v); inserted=True
                proposal['derived_proposal']=derived
            else: proposal_reason=proposal_reason or 'no_local_justified_intervention_model_report'
        write_json(out/api.role/'proposals'/(p['audit_id']+'.json'), proposal)
        for n in range(len(ro['steps'])+1, 10):
            if ro['termination']!='not_finished': break
            write_json(out/api.role/'checkpoints'/(p['audit_id']+'.json'), dict(rollout=ro,next_step=n))
            parsed = step(ro, api, p, n, 's'+str(n))
            if parsed and parsed['kind']=='grounding':
                v, meta = old.execute_crop(native, ro['views'], parsed, n, odir)
                ro['steps'][-1].update(meta)
                if any(x['effective_box_px']==v['effective_box_px'] for x in ro['views']): ro['steps'][-1]['flags'].append('repeat_crop')
                ro['views'].append(v)
    if ro['termination']=='not_finished': ro['termination']='budget_exhausted'
    ro['crops']=len(ro['views'])-1
    if ro['termination'] in ('answer','abstain'): ro['export']=old.student_export(ro,p['question'])
    violations=[x for s in ro['steps'] for x in s.get('violations', [])]
    violations += [f for s in ro['steps'] for f in s.get('flags', []) if f in ('coordinate_talk','coordinate_unresolved')]
    passed=ro['termination']=='answer' and not violations
    gt=score(private['answer_spec'],ro.get('answer') or '')
    mask_ok=bool(ro['export']) and ro['export']['assistant_supervision']==[s['origin']!='editor_forced' for s in ro['steps']]
    eligible=bool(passed and gt['score']==1 and mask_ok)
    # Originally-direct tasks may correctly remain direct. Originally-tool tasks
    # continue down the construction funnel when no intervention was feasible.
    needs_fallback=not eligible or (p['original_crops']>0 and not inserted)
    if not eligible: reason='objective_failure:'+ro['termination'] if not passed else 'GT_or_flags_conflict'
    elif needs_fallback: reason='tool_task_intervention_not_constructed'
    else: reason=None
    r=dict(audit_id=p['audit_id'],item_id=p['item_id'],model=api.model,assigned_role=api.role,
           effort='low',original_model=p['original_model'],original_crops=p['original_crops'],
           rollout=ro,bootstrap=bootstrap,prefix_origin=prefix_origin,interpolation_inserted=inserted,
           intervention_not_applicable_reason=proposal_reason,paired_split=private['split'],
           semantic_review='NOT_REQUESTED_BY_USER',process_quality_unverified=True,
           cpu=dict(passed=passed,reasons=violations),
           objective_checks=dict(protocol_passed=passed,GT_score=gt,editor_supervision_flags_correct=mask_ok,
               eligible_without_process_review=eligible,actual_token_mask_verified=False,target_4B_loader_verified=False),
           needs_fallback=needs_fallback,fallback_reason=reason,config_sha256=old.sha(out/'config.json'),
           provenance='fresh first decision; fresh crop preferred; old visible first prefix only when fresh direct on originally-tool task; one forced masked intervention; autonomous continuation')
    write_json(out/api.role/'results'/(p['audit_id']+'.json'),r)
    return r


def pending_requests(parent, role):
    out=parent/role; dispatch=old.read_rows(out/'budget/dispatch.jsonl') if (out/'budget/dispatch.jsonl').exists() else []
    pending=[]
    for d in dispatch:
        ap=out/role/'attempts'/(d['request_sha256']+'.json')
        if not ap.exists(): pending.append(d['request_sha256']); continue
        a=read(ap)
        if a.get('state')!='settled' or not Path(a.get('receipt','')).is_file() or old.sha(a['receipt'])!=a['receipt_sha256']:
            pending.append(d['request_sha256'])
    return pending


def choose_ids(parent, role):
    ids=[r['audit_id'] for r in read(parent/'SCOPE_ALL_4040.json')]
    if role=='sol': return ids
    previous=ROLES[ROLES.index(role)-1]
    out=parent/previous
    confirmed=[]
    for aid in ids:
        path=out/previous/'results'/(aid+'.json')
        if not path.exists():continue
        r=read(path)
        if not r.get('needs_fallback'):continue
        keys={s['request_sha256'] for ro in (r['rollout'],r.get('bootstrap',{})) for s in ro.get('steps',[]) if s.get('request_sha256')}
        keys.update(r.get('episode_request_sha256s',[]))
        proposal=out/previous/'proposals'/(aid+'.json')
        if proposal.exists():keys.add(read(proposal)['request_sha256'])
        known=True
        for key in keys:
            attempt=out/previous/'attempts'/(key+'.json');receipt=out/previous/'calls'/(key+'.json')
            if not attempt.exists() or not receipt.exists():known=False;break
            a=read(attempt);reply=read(receipt)
            if a.get('state')!='settled' or a.get('receipt_sha256')!=old.sha(receipt):known=False;break
            if any(e.get('error_type') in ('ReadTimeout','ConnectTimeout','ConnectionError','Timeout') for e in reply.get('errors',[])):
                known=False;break
        if known and keys:confirmed.append(aid)
    return confirmed


def run(parent, role):
    cfg=verify(parent,role);out=parent/role
    with (out/'RUN.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'RUN_STATE.json').exists(): raise ValueError('Never restart unknown/issued branch')
        ledger=Ledger(out/'budget',{s['model']:s['call_cap'] for s in cfg['models'].values()})
        api=Transport(out,role,cfg,ledger);private={r['audit_id']:r for r in read(parent/'SELECTION_PRIVATE.json')}
        started=set();lock=threading.Lock();first_ok=False
        sol_ids=[r['audit_id'] for r in read(parent/'SCOPE_ALL_4040.json')];cursor=0;fallback_ids=[];last_scan=0
        write_json(out/'RUN_STATE.json',dict(state='running',started_at=now(),pid=os.getpid()))
        def one(aid):
            try:
                api.check();p=packet(parent,aid)
                r=generate(parent,p,api,private[aid])
                print(json.dumps(dict(audit_id=aid,model=api.model,termination=r['rollout']['termination'],
                    interpolation=r['interpolation_inserted'],eligible=r['objective_checks']['eligible_without_process_review'],fallback=r['needs_fallback'])),flush=True)
                return r
            except Exception as ex:
                write_json(out/'task_errors'/(aid+'.json'),dict(audit_id=aid,error_type=type(ex).__name__,at=now()))
                dispatch=ledger.rows();keys=[d['request_sha256'] for d in dispatch if d['label'].split('/')[0]==aid]
                if keys:
                    # Preserve an interrupted episode as a technical disposition,
                    # not a model judgment. Per-request settlement gates fallback.
                    cp=out/api.role/'checkpoints'/(aid+'.json')
                    ro=read(cp)['rollout'] if cp.exists() else dict(steps=[],views=[],crops=0,answer=None,export=None)
                    ro.update(termination='episode_interrupted',export=None,crops=max(0,len(ro.get('views',[]))-1))
                    r=dict(audit_id=aid,item_id=private[aid]['manifest']['item_id'],model=api.model,assigned_role=role,
                           rollout=ro,bootstrap={},interpolation_inserted=any(s.get('origin')=='editor_forced' for s in ro['steps']),
                           episode_request_sha256s=keys,prefix_origin='incomplete_technical_episode',
                           paired_split=private[aid]['split'],semantic_review='NOT_REQUESTED_BY_USER',process_quality_unverified=True,
                           cpu=dict(passed=False,reasons=[type(ex).__name__]),
                           objective_checks=dict(eligible_without_process_review=False,protocol_passed=False,GT_score={'score':None},
                               editor_supervision_flags_correct=False,actual_token_mask_verified=False,target_4B_loader_verified=False),
                           needs_fallback=True,fallback_reason='episode_interrupted:'+type(ex).__name__)
                    write_json(out/api.role/'results'/(aid+'.json'),r)
                ledger.stop(api.model,'Episode exception; no same-role retry')
                return None
        def next_id():
            nonlocal cursor,fallback_ids,last_scan
            with lock:
                if role=='sol':
                    if cursor>=len(sol_ids):return None
                    aid=sol_ids[cursor];cursor+=1;started.add(aid);return aid
                if time.monotonic()-last_scan>=5:
                    fallback_ids=choose_ids(parent,role);last_scan=time.monotonic()
                for aid in fallback_ids:
                    if aid not in started: started.add(aid); return aid
            return None
        def previous_done():
            if role=='sol': return True
            pre=parent/ROLES[ROLES.index(role)-1]/'RUN_STATE.json'
            return pre.exists() and read(pre)['state']!='running'
        # The first complete real task gates high concurrency, including errors
        # later in its image/context chain (the historical Sol s5 failure).
        while True:
            try:api.check()
            except BudgetStop:break
            aid=next_id()
            if aid:
                result=one(aid)
                first_ok=result is not None and result['rollout']['termination'] not in ('api_error','editor_transport_error')
                if not first_ok: ledger.stop(api.model,'First real transport/input failure')
                else:
                    receipts=[read(p) for p in (out/role/'calls').glob('*.json')]
                    write_json(out/'FIRST_RECEIPT_CHECK.json',dict(at=now(),task=aid,real_calls=len(receipts),
                        returned_models=dict(Counter(c.get('returned_model') for c in receipts)),
                        actual_input_hashes=[c['images'] for c in receipts],errors=sum(c['finish_reason']=='error' for c in receipts),
                        first_task_complete=True,process_quality_verified=False))
                break
            if previous_done(): break
            time.sleep(5)
        def worker():
            while first_ok:
                try:api.check()
                except BudgetStop:return
                aid=next_id()
                if aid:one(aid)
                elif previous_done():return
                else:time.sleep(5)
        if first_ok:
            with ThreadPoolExecutor(max_workers=cfg['models'][role]['workers']) as pool:
                fs=[pool.submit(worker) for _ in range(cfg['models'][role]['workers'])]
                for f in fs:f.result()
        saved=len(list((out/role/'results').glob('*.json')))
        stop=(out/'budget'/('STOP_'+api.model)).exists() or (out/'PAUSE').exists() or (parent/'PAUSE').exists()
        write_json(out/'RUN_STATE.json',dict(state='stopped_incomplete' if stop else 'generation_complete_unreviewed',
                  finished_at=now(),started_tasks=len(started),saved_results=saved,pending=pending_requests(parent,role)))


def start_gemini_proxy(parent):
    out=parent/'gemini'
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',4003))==0:raise ValueError('4003 occupied; do not touch other services')
    config=out/'gemini_litellm.yaml'
    config.write_text('''model_list:
  - model_name: gemini-3.8-flash
    litellm_params:
      model: openai/gemini-3.8-flash
      api_base: http://127.0.0.1:8317/v1
      api_key: os.environ/ICML_SOL_FULL_GEMINI_UPSTREAM
      num_retries: 0
general_settings:
  master_key: os.environ/ICML_SOL_FULL_LITELLM_KEY
litellm_settings:
  drop_params: true
  num_retries: 0
router_settings:
  num_retries: 0
''')
    tm=old.transport.base.routing.tm
    key=dotenv_values(tm.REPO_ENV).get('LITELLM_API_KEY');upstream=dotenv_values(tm.PROXY_ENV).get('CLIPROXY_API_KEY')
    if not key or not upstream:raise ValueError('Missing existing credentials')
    env=dict(os.environ,ICML_SOL_FULL_GEMINI_UPSTREAM=upstream,ICML_SOL_FULL_LITELLM_KEY=key,
             LITELLM_LOCAL_MODEL_COST_MAP='true',DO_NOT_TRACK='true',CUDA_VISIBLE_DEVICES='')
    command=['${PROXY_DIR}/litellm/.venv/bin/litellm','--config',str(config),
             '--host','127.0.0.1','--port','4003','--num_workers','1']
    with (out/'gemini_litellm.log').open('ab') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'PROXY_LAUNCH.json',dict(pid=proc.pid,command=command,at=now(),config_sha256=old.sha(config)))
    for _ in range(45):
        if proc.poll() is not None:break
        try:
            r=requests.get('http://127.0.0.1:4003/v1/models',headers={'Authorization':'Bearer '+key},timeout=2)
            if r.status_code==200 and [x['id'] for x in r.json().get('data',[])]==['gemini-3.8-flash']:return
        except requests.RequestException:pass
        time.sleep(1)
    proc.terminate();proc.wait(timeout=10)
    raise RuntimeError('Batch Gemini proxy not ready; zero model requests')


def launch(parent,role):
    cfg=verify(parent,role);out=parent/role
    if any((out/n).exists() for n in ('LAUNCH.json','RUN_STATE.json','PAUSE')):raise ValueError('Fresh launch only')
    if (parent/'PAUSE').exists():raise ValueError('Parent paused')
    if role=='gemini':
        if time.time()<datetime.fromisoformat(cfg['gemini_not_before_utc']).timestamp():raise ValueError('Known Gemini quota window not elapsed')
        if not choose_ids(parent,role):raise ValueError('No confirmed Gemini fallback work')
        start_gemini_proxy(parent)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    cmd=['${PYTHON}','-u',str(Path(__file__).resolve()),'run','--parent',str(parent),'--role',role]
    with (out/'worker.log').open('ab') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'LAUNCH.json',dict(pid=proc.pid,command=cmd,at=now(),workers=cfg['models'][role]['workers'],rpm=cfg['models'][role]['rpm'],
                                    real_first_task_verified=False))
    return dict(launched=True,role=role,workers=cfg['models'][role]['workers'],first_real_receipt_pending=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','verify','launch','run']);ap.add_argument('--parent',type=Path,default=DEFAULT_PARENT);ap.add_argument('--role',choices=ROLES)
    args=ap.parse_args();parent=args.parent.resolve()
    if args.action=='prepare':result=prepare(parent)
    elif args.action=='verify':result=verify(parent,args.role)
    elif args.action=='launch':result=launch(parent,args.role)
    else:result=run(parent,args.role)
    print(json.dumps(result,ensure_ascii=False))
