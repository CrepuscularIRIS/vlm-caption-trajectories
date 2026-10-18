"""Full train-tool interpolation, separate from immutable pilot runs.

Only a visible prefix reaches the editor/continuation. One proposed intervention,
one fresh continuation, no retries, no automatic semantic KEEP or promotion.
"""
import argparse
import base64
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import html
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

import caption_v51 as proto
import interp_recovery_calibration as old
import interp_recovery_v2 as pilot
from agy_reader import parse_agy
from probe_api import digest, structured, text_msg
from tiered_transport import BudgetStop, Ledger, ROOT, write_json

VERSION = 'interp-recovery-full-prefix-editor-20261011'
OVERLAY = ROOT/'obs/runs/interp_recovery_v2_20261011/GROUP_SPLIT_OVERLAY.json'
PILOT_PRIVATE = ROOT/'obs/runs/interp_recovery_v2_20261011/SELECTION_PRIVATE.json'
PILOT_REVIEW = ROOT/'reports/interp_recovery_v2_20261011/records/P008.json'

EDITOR_SYSTEM = '''You propose ONE local visual intervention, not an answer or an evaluator judgment.
You receive only a question, a completed first assistant crop action, the original
displayed image and its returned observation_1. No future trajectory or answer is available.
Describe only currently supported facts, instance bindings and genuine uncertainty
in a short English caption (1-3 sentences); cite [original_image] or [observation_1].
Do not invent uncertainty or say a later crop succeeded/failed. Keep current facts;
do not include a new action plan, answer letter, coordinates or future observations in caption.
Propose a plausible local framing mistake: tight_context removes nearby context, or
local_shift slightly offsets the framing near the same candidate. Prefer source
observation_1; local_shift may use original_image near the first crop. This is an
editor-inserted action, not a demonstration of correct localization. Geometry does
not certify semantic failure. Never use unrelated distant filler regions.
If a justified local intervention cannot be proposed, set applicable=false and explain.
Do not require more crops or five steps; continuation will decide from all old evidence.
Return only JSON with exactly these keys:
{"applicable":true,"caption":"...","kind":"tight_context",
 "source":"observation_1","bbox":[0.2,0.2,0.7,0.7],
 "expected_missing_evidence":"specific detail or context this framing might omit",
 "reason":"why this is a plausible local framing mistake from the visible prefix"}.
For applicable=false, kind/source/bbox may be null. No GT, answer or future-box fields.'''


def make_prefix(result, row, aid, role):
    """Allowlist independent of old answer, next Caption/action and future views."""
    ro = result['rollout']
    turns = [c['value'] for c in ro['export']['conversations'] if c['from']=='gpt']
    first = proto.parse_step(turns[0])
    if first['kind']!='grounding' or proto.check_step(first,1,0,['original_image'],8):
        raise ValueError('Illegal original first crop')
    packet = dict(audit_id=aid,item_id=row['item_id'],question=result['item']['question'],
                  image=result['item']['image'],source_image_sha256=old.sha(result['item']['image']),
                  assigned_role=role,original_model=row['model'],prefix_assistant=turns[0],
                  prefix_views=copy.deepcopy(ro['views'][:2]),budget=8)
    if [v['view_id'] for v in packet['prefix_views']]!=['original_image','observation_1']:
        raise ValueError('Only the first actually received crop is allowed')
    return packet


def allocate(rows):
    """70/30 within light source/material/crop strata, no outcome-based routing."""
    strata=defaultdict(list)
    for r in rows:
        strata[r['bench'],r['material'],1 if r['crops']==1 else 2].append(r)
    assigned={}
    for key, pool in sorted(strata.items()):
        pool.sort(key=lambda r:digest([VERSION,r['canonical_sample_key']]))
        # Local rounding is deterministic and recorded, never a quality quota.
        nk=int(math.floor(len(pool)*.3+.5))
        for i,r in enumerate(pool):
            assigned[r['canonical_sample_key']]='k3' if i<nk else 'gemini'
    return assigned


def prepare(out):
    if out.exists():raise ValueError('Fresh run required')
    rows=old.read_rows(old.BASELINE)
    if len(rows)!=4040 or len({r['canonical_sample_key'] for r in rows})!=4040:
        raise ValueError('Baseline cardinality/uniqueness changed')
    overlay=json.loads(OVERLAY.read_text())
    if any(len(r['canonical_groups'])!=1 for r in rows):
        raise ValueError('Multi-group task needs linked split handling')
    prior=json.loads(PILOT_REVIEW.read_text())
    if prior['verdict']!='KEEP' or not prior['recovery_selected']:
        raise ValueError('Historical approved replacement changed')
    reused=next(p for p in json.loads(PILOT_PRIVATE.read_text()) if p['audit_id']=='P008')
    reused_key=reused['manifest']['canonical_sample_key']
    scope=[]; selected=[]
    for r in rows:
        split=overlay['groups'][r['canonical_groups'][0]]
        if split!='train':reason='retain_heldout_original'
        elif r['crops']==0:reason='retain_direct_original'
        elif r['crops']>=8:reason='retain_at_crop_budget_limit'
        elif r['canonical_sample_key']==reused_key:reason='reuse_approved_P008_not_regenerate'
        else:reason='queue_new_prefix_only_interpolation';selected.append(r)
        scope.append(dict(canonical_sample_key=r['canonical_sample_key'],baseline_id=r['audit_id'],
                          item_id=r['item_id'],original_model=r['model'],groups=r['canonical_groups'],
                          split=split,crops=r['crops'],disposition=reason))
    assigned=allocate(selected);out.mkdir(parents=True)
    write_json(out/'GROUP_SPLIT_OVERLAY.json',overlay)
    write_json(out/'SCOPE_ALL_4040.json',scope)
    write_json(out/'REUSE_APPROVED.json',dict(audit_id='P008',canonical_sample_key=reused_key,
               manifest=reused['manifest'],review_path=str(PILOT_REVIEW),review_sha256=old.sha(PILOT_REVIEW),
               result_path=str(ROOT/'obs/runs/interp_recovery_v2_20261011/k3/results/P008.json'),
               result_sha256=prior['source_result_sha256'],requires_target_4B_loader=True))
    packets=[];private=[];hashes={}
    selected.sort(key=lambda r:digest([VERSION,'order',r['canonical_sample_key']]))
    for i,r in enumerate(selected,1):
        src=Path(r['result_path'])
        if old.sha(src)!=r['result_sha256']:raise ValueError('Old result changed')
        result=json.loads(src.read_text());aid=f'B{i:04d}'
        p=make_prefix(result,r,aid,assigned[r['canonical_sample_key']])
        for v in p['prefix_views']:
            if old.sha(v['path'])!=v['sha256']:raise ValueError('Old prefix pixels changed')
        path=out/'inputs'/(aid+'.json');write_json(path,p);hashes[str(path)]=old.sha(path);packets.append(p)
        private.append(dict(audit_id=aid,manifest=r,answer_spec=result['item']['answer_spec'],
                            source_effort=result.get('effort'),assigned_role=p['assigned_role']))
    write_json(out/'INPUT_HASHES.json',hashes)
    write_json(out/'SELECTION_PRIVATE.json',private)
    counts=Counter(p['assigned_role'] for p in packets)
    files=set(old.transport.base.CODE_FILES)|{'interp_recovery_full.py','interp_recovery_v2.py',
            'interp_recovery_calibration.py','sft_v2_topup.py','agy_reader.py','export_protocol_check.py','caption_v3.py'}
    frozen={str(ROOT/'obs/src'/f):old.sha(ROOT/'obs/src'/f) for f in files}
    for path in (old.BASELINE,OVERLAY,PILOT_REVIEW,PILOT_PRIVATE,out/'INPUT_HASHES.json',
                 out/'SELECTION_PRIVATE.json',out/'SCOPE_ALL_4040.json',out/'REUSE_APPROVED.json',out/'GROUP_SPLIT_OVERLAY.json'):
        frozen[str(path)]=old.sha(path)
    cfg=dict(version=VERSION,created_at=time.time(),deadline=time.time()+48*3600,
             authorization='2026-10-11 user: 继续做; 按照两个设计; 这次直接全部构造; SFT让学会纠正行为',
             tasks=len(packets),workers=4,budget_crops=8,examples=[],frozen_files=frozen,
             baseline_manifest=str(old.BASELINE),baseline_T_Q_G=[4040,4040,3690],
             scope='all 1..7-crop train-overlay tasks; one approved P008 reused; direct and heldout unchanged',
             editor='one model proposal from only actual first-step prefix; NOT visually certified before review',
             continuation='K3/Gemini fresh continuation, original-prefix teacher separately recorded; no old suffix/GT',
             intervention='one prefix-local/tight framing proposal; no future evidence box, no forced five-step count',
             retries=dict(answer=0,format=0,transport=0,proposal=0),
             masks='editor assistant body+EOS=0; prefix and continuation eligible only after full process review',
             source_checks='global frozen manifests/code each call; selected input and source pixels at dispatch',
             no_gpu=True,no_training=True,no_download=True,no_old_run_resume=True,
             models={role:dict(model='gemini-3.8-flash-low' if role=='gemini' else 'k3-256k',
                effort='low',route='agy' if role=='gemini' else 'api',
                endpoint=old.transport.AGY if role=='gemini' else 'http://127.0.0.1:4001',
                tasks=counts[role],workers=2,rpm=12,call_cap=counts[role]*8,max_tokens=8192,
                allowed_returned_models=['gemini-3.8-flash-low' if role=='gemini' else 'k3-256k']) for role in ('gemini','k3')},
             budget_units='one editor + at most seven continuation requests per task; AGY CLI / K3 HTTP; internal AGY requests unobservable')
    write_json(out/'config.json',cfg);(out/'CONFIG.sha256').write_text(old.sha(out/'config.json')+'\n')
    stats=dict(baseline_T_Q_G=[4040,4040,3690],total_tool_tasks=1908,new_tasks=len(packets),reused=1,
        scope_counts=dict(Counter(s['disposition'] for s in scope)),allocation=dict(counts),
        old_crops=dict(Counter(p['manifest']['crops'] for p in private)),
        source_models=dict(Counter(p['manifest']['model'] for p in private)),
        benchmarks=dict(Counter(p['manifest']['bench'] for p in private)),
        materials=dict(Counter(p['manifest']['material'] for p in private)),
        new_unique_groups=len({g for p in private for g in p['manifest']['canonical_groups']}))
    write_json(out/'SELECTION_STATS.json',stats)
    return status(out)


def verify(out, pixels=False):
    cfg=json.loads((out/'config.json').read_text())
    if cfg['version']!=VERSION or old.sha(out/'config.json')!=(out/'CONFIG.sha256').read_text().strip():
        raise ValueError('Frozen configuration changed')
    for path,h in cfg['frozen_files'].items():
        if old.sha(path)!=h:raise ValueError('Frozen source changed: '+path)
    hashes=json.loads((out/'INPUT_HASHES.json').read_text());packets=[]
    for path,h in sorted(hashes.items()):
        if old.sha(path)!=h:raise ValueError('Input changed')
        p=json.loads(Path(path).read_text());packets.append(p)
        if pixels:verify_packet(p)
    if len(packets)!=cfg['tasks']:raise ValueError('Task inventory mismatch')
    return cfg,packets


def verify_packet(packet):
    if old.sha(packet['image'])!=packet['source_image_sha256']:raise BudgetStop('Native source image changed')
    for v in packet['prefix_views']:
        if old.sha(v['path'])!=v['sha256']:raise BudgetStop('Prefix view changed')


def editor_messages(packet, api):
    content=[dict(type='text',text='Question:\n'+packet['question']+'\nCompleted Step 1:\n'+packet['prefix_assistant'])]
    for v in packet['prefix_views']:
        content.extend([dict(type='text',text='['+v['view_id']+']'),api.image(v['path'])])
    return [text_msg('system',EDITOR_SYSTEM),dict(role='user',content=content)]


def validate_proposal(obj, packet):
    keys={'applicable','caption','kind','source','bbox','expected_missing_evidence','reason'}
    if not isinstance(obj,dict) or set(obj)!=keys or type(obj['applicable']) is not bool:
        raise ValueError('Invalid editor JSON schema')
    if not isinstance(obj['reason'],str) or not obj['reason'].strip():raise ValueError('Missing intervention reason')
    if not obj['applicable']:return None
    if obj['kind'] not in ('tight_context','local_shift'):raise ValueError('Invalid intervention kind')
    if obj['source'] not in ('original_image','observation_1'):raise ValueError('Future/unavailable source')
    if obj['kind']=='tight_context' and obj['source']!='observation_1':raise ValueError('Tight crop must use current crop')
    b=obj['bbox']
    if not isinstance(b,list) or len(b)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in b):
        raise ValueError('Invalid bbox numbers')
    if not (0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):raise ValueError('Invalid bbox range')
    area=(b[2]-b[0])*(b[3]-b[1])
    if not .015<=area<=.65:raise ValueError('Unreasonable local crop area')
    if obj['source']=='original_image':
        f=packet['prefix_views'][1]['fov_original'];w=f[2]-f[0];h=f[3]-f[1]
        cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        if not(f[0]-.5*w<=cx<=f[2]+.5*w and f[1]-.5*h<=cy<=f[3]+.5*h):
            raise ValueError('Distant filler crop rejected')
    cap=obj['caption']
    if not isinstance(cap,str) or not 15<=len(cap)<=700 or not re.search(r'\[(original_image|observation_1)\]',cap):
        raise ValueError('Missing short anchored fact caption')
    if re.search(r'\[(?!original_image\]|observation_1\])|<|>|\bStep\s*\d|\b(?:ground\s*truth|correct\s*answer)\b',cap,re.I):
        raise ValueError('Future/answer/protocol leakage in caption')
    if not isinstance(obj['expected_missing_evidence'],str) or not obj['expected_missing_evidence'].strip():
        raise ValueError('No specific evidence-gap hypothesis')
    p=copy.deepcopy(packet)
    p.update(intervention=obj['kind'],perturbation=dict(source=obj['source'],bbox=b),fact_caption=cap,
             proposal_status='NOT_VISUALLY_VERIFIED',proposal_reason=obj['reason'],
             expected_missing_evidence=obj['expected_missing_evidence'])
    parsed=pilot.forced_turn(p)
    if 'coordinate_talk' in proto.soft_flags(parsed):raise ValueError('Coordinates in Caption')
    return p


class FullTransport(old.RecoveryTransport):
    def agy_call(self,msgs,key):
        """Fresh staged-only CLI; explicitly reread all images on every target turn."""
        wd=Path('/tmp')/('icml_interp_full_'+key);wd.mkdir(exist_ok=False)
        parts=[];staged=[]
        for msg in msgs:
            parts.append('\nMESSAGE ROLE: '+msg['role'])
            c=msg['content']
            for b in ([dict(type='text',text=c)] if isinstance(c,str) else c):
                if b['type']=='text':parts.append(b['text'])
                elif b['type']=='image_url':
                    p=wd/f'image_{len(staged):02d}.jpg'
                    p.write_bytes(base64.b64decode(b['image_url']['url'].split(',',1)[1]));staged.append(p)
                    parts.append('IMAGE ATTACHMENT: '+str(p))
                else:raise ValueError('Unsupported message block')
        checklist='\n'.join(str(p) for p in staged)
        prompt=('This invocation has NO visual memory of earlier invocations. FIRST open EVERY path below, in order, '
                'with view_file, including image_00 (the original image) even if a later crop seems sufficient. '
                'A response with even one unread image is rejected. Only view_file and finish are allowed; '
                'no commands, network, other files or delegates.\nMANDATORY READ CHECKLIST:\n'+checklist+
                '\nAfter all reads, follow the SYSTEM instructions in the supplied transcript. '
                'If they request editor JSON, return exactly that JSON; otherwise continue exactly ONE assistant '
                'Caption/action step, output the grounding action instead of executing it.\n'+ '\n'.join(parts))
        cmd=[old.transport.AGY,'-p',prompt,'--model',self.model,'--effort',self.spec['effort'],
             '--output-format','stream-json','--sandbox','--disable-slash-commands','--new-project',
             '--print-timeout','5m0s','--log-file',str(self.out/'raw'/(key+'.agy.log'))]
        write_json(self.out/'raw'/(key+'.request.json'),dict(prompt=prompt,command=cmd[:1]+['-p','<prompt>']+cmd[3:],cwd=str(wd)))
        proc=subprocess.Popen(cmd,cwd=wd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,start_new_session=True)
        try:stdout,stderr=proc.communicate(timeout=330);code=proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:stdout,stderr=proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate()
            code='timeout'
        (self.out/'raw'/(key+'.stdout.jsonl')).write_text(stdout)
        (self.out/'raw'/(key+'.stderr.txt')).write_text(stderr)
        meta=parse_agy(stdout,staged,self.model)
        # A DONE event accompanied by a tool error is not a successful image read.
        bad_reads=[]
        for line in stdout.splitlines():
            try:e=json.loads(line)
            except ValueError:continue
            s=e.get('step_update') or {}
            if s.get('tool_name')=='view_file' and s.get('state') in ('ERROR','FAILED'):
                bad_reads.append(s.get('step_index'))
            if s.get('tool_name')=='view_file' and any(s.get(k) for k in ('error','is_error')):
                bad_reads.append(s.get('step_index'))
        meta['tool_read_errors']=bad_reads
        valid=meta['valid'] and code==0 and not bad_reads
        if re.search(r'quota|rate.?limit|unauth|authentication|\b429\b|credits.exhaust',stdout+'\n'+stderr,re.I):
            self.ledger.stop(self.model,'AGY quota/auth/rate marker')
        if meta['returned_models'] and meta['returned_models']!=[self.model]:self.ledger.stop(self.model,'AGY identity mismatch')
        # Missing-image output is a semantic input failure: stop branch, never just keep going.
        if not meta['all_images_opened'] or meta['disallowed_tools'] or meta['unexpected_paths'] or bad_reads:
            self.ledger.stop(self.model,'AGY image-read/tool scope failure')
        return dict(content=meta['text'],finish_reason='stop' if valid else 'error',
                    returned_model=meta['returned_models'][0] if meta['returned_models'] else None,
                    usage={},errors=[] if valid else [dict(error_type='agy_invalid',exit=code)],
                    agy_receipt=meta,internal_server_request_count=None,workdir=str(wd))


def generate(out,packet,api):
    verify_packet(packet)
    reply=api.call(packet['audit_id']+'/editor',editor_messages(packet,api),
                   replicate_id=VERSION+'#editor#'+packet['audit_id'],temperature=.3,top_p=.9)
    proposal=dict(audit_id=packet['audit_id'],request_sha256=reply['request_sha256'],
                  input_sha256=old.sha(out/'inputs'/(packet['audit_id']+'.json')),
                  raw_text=reply.get('content') or '',finish_reason=reply['finish_reason'])
    write_json(out/api.role/'proposals'/(packet['audit_id']+'.json'),proposal)
    if reply['finish_reason']!='stop':
        outcome='editor_transport_error' if reply['finish_reason']=='error' else 'editor_provider_finish_error'
        derived=None
    else:
        try:derived=validate_proposal(structured(reply.get('content') or ''),packet)
        except ValueError as exc:derived=None;outcome='editor_schema_or_geometry_rejected';proposal['rejection_reason']=str(exc)
        else:outcome='no_justified_intervention' if derived is None else 'proposal_accepted_pending_visual_review'
    proposal['outcome']=outcome;write_json(out/api.role/'proposals'/(packet['audit_id']+'.json'),proposal)
    if derived is None:
        result=dict(audit_id=packet['audit_id'],item_id=packet['item_id'],assigned_role=api.role,model=api.model,
                    original_model=packet['original_model'],editor_request_sha256=reply['request_sha256'],
                    rollout=dict(termination=outcome,crops=1,steps=[],views=[],answer=None,export=None),
                    cpu=dict(passed=False,reasons=[outcome]),semantic_review='NOT_REVIEWED',
                    config_sha256=old.sha(out/'config.json'))
    else:
        write_json(out/api.role/'curated_inputs'/(packet['audit_id']+'.json'),derived)
        result=pilot.generate(out,derived,api)
        result.update(original_model=packet['original_model'],editor_request_sha256=reply['request_sha256'],
                      proposal_status='NOT_VISUALLY_VERIFIED',
                      provenance='original teacher first-step prefix + model-proposed masked editor intervention + fresh '+api.role+' continuation; no old suffix')
    write_json(out/api.role/'results'/(packet['audit_id']+'.json'),result)
    if result['cpu']['passed']:review_packet(out,packet,result)
    return result


def review_packet(out,packet,result):
    """Neutral review input; GT/old conclusions live only in private selection."""
    ro=result['rollout'];aid=packet['audit_id']
    record=dict(audit_id=aid,question=packet['question'],
        source_result=str(out/result['assigned_role']/'results'/(aid+'.json')),
        source_result_sha256=old.sha(out/result['assigned_role']/'results'/(aid+'.json')),
        proposal_status='NOT_VISUALLY_VERIFIED',
        views=[{k:v[k] for k in ('view_id','parent','path','sha256','source_px','shown_px','fov_original') if k in v} for v in ro['views']],
        steps=[dict(step=s['step'],text=s['parsed']['text'],origin=s['origin'],
                    available_before=s['available_before'],observation=s.get('observation')) for s in ro['steps']],
        review_order='Open original and EVERY returned crop; check prefix facts and editor proposal before scoring recovery; process record first, GT reconciliation later.')
    write_json(out/'review_packets'/(aid+'.json'),record)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+aid+' 插值复核</title>',
          '<style>body{max-width:1100px;margin:2rem auto;font:16px system-ui}img{max-width:100%;max-height:80vh}pre{white-space:pre-wrap;background:#eee;padding:1rem}</style>',
          '<h1>'+aid+'｜尚未过程验收</h1><p>'+html.escape(packet['question'])+'</p>',
          '<p>先看全部实际图片，再核事实、实例绑定、编辑轮是否真实缺证据、续写是否承认并修复。强插轮不监督。GT和历史结论不在此页。</p>']
    for v in ro['views']:
        body.extend(['<h2>'+html.escape(v['view_id'])+'</h2><img src="'+html.escape(v['path'],quote=True)+'">'])
    for s in record['steps']:
        body.extend(['<h2>Step '+str(s['step'])+'｜'+s['origin']+'</h2><pre>'+html.escape(s['text'])+'</pre>'])
    (out/'review_packets'/(aid+'.html')).write_text('\n'.join(body)+'\n</html>')


def status(out):
    cfg=json.loads((out/'config.json').read_text());by={}
    issued=old.read_rows(out/'budget/dispatch.jsonl') if (out/'budget/dispatch.jsonl').exists() else []
    for role,s in cfg['models'].items():
        results=[json.loads(p.read_text()) for p in (out/role/'results').glob('*.json')]
        calls=[json.loads(p.read_text()) for p in (out/role/'calls').glob('*.json')]
        dispatch=[r for r in issued if r['model']==s['model']];settled={c['request_sha256'] for c in calls}
        started={r['label'].split('/')[0] for r in dispatch}
        by[role]=dict(planned=s['tasks'],saved=len(results),unattempted=s['tasks']-len(started),
            started_tasks_without_saved_result=len(started)-len(results),without_saved_result=s['tasks']-len(results),
            cpu_candidates=sum(r['cpu']['passed'] for r in results),
            terminations=dict(Counter(r['rollout']['termination'] for r in results)),
            crops=dict(Counter(r['rollout']['crops'] for r in results)),dispatched=len(dispatch),settled=len(calls),
            pending=[r['request_sha256'] for r in dispatch if r['request_sha256'] not in settled],
            returned_models=dict(Counter(str(r.get('returned_model')) for r in calls)),
            request_errors=sum(r['finish_reason']=='error' for r in calls),
            last_receipt_mtime=max((p.stat().st_mtime for p in (out/role/'calls').glob('*.json')),default=None),
            stopped=(out/'budget'/('STOP_'+s['model'])).exists())
    reviews=[json.loads(p.read_text()) for p in (out/'reviews').glob('*.json')]
    state=json.loads((out/'RUN_STATE.json').read_text()) if (out/'RUN_STATE.json').exists() else dict(state='prepared')
    data=dict(updated_at=time.time(),run_state=state,planned=cfg['tasks'],saved=sum(x['saved'] for x in by.values()),
              generators=by,process_reviews=dict(Counter(r['verdict'] for r in reviews)),
              not_process_reviewed=sum(v['cpu_candidates'] for v in by.values())-len(reviews),
              approved_historical_variants=1,baseline_unchanged_T_Q_G=[4040,4040,3690],
              generation_is_not_SFT_acceptance=True)
    write_json(out/'STATUS.json',data);return data


def run(out):
    cfg,packets=verify(out,pixels=True)
    # Reuse tested rollout code with this batch's request namespace, not pilot ids.
    pilot.VERSION=VERSION
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'RUN_STATE.json').exists():raise ValueError('No automatic restart; reconcile attempts/lock before user-authorized resume')
        ledger=Ledger(out/'budget',{s['model']:s['call_cap'] for s in cfg['models'].values()})
        apis={r:FullTransport(out,r,cfg,ledger) for r in cfg['models']}
        hashes=json.loads((out/'INPUT_HASHES.json').read_text())
        queues={r:deque(p for p in packets if p['assigned_role']==r) for r in apis}
        write_json(out/'RUN_STATE.json',dict(state='running',pid=os.getpid(),started_at=time.time()))
        done=threading.Event();qlock=threading.Lock();errors=[]
        def watch():
            while not done.wait(30):status(out)
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        def one(role,p):
            try:
                apis[role].check()
                path=out/'inputs'/(p['audit_id']+'.json')
                if old.sha(path)!=hashes[str(path)]:
                    write_json(out/'PAUSE',dict(reason='input_changed',audit_id=p['audit_id']));raise BudgetStop('Input changed')
                try:verify_packet(p)
                except BudgetStop:
                    write_json(out/'PAUSE',dict(reason='source_pixels_changed',audit_id=p['audit_id']));raise
                r=generate(out,p,apis[role])
                print(json.dumps(dict(audit_id=r['audit_id'],role=role,termination=r['rollout']['termination'],
                                     crops=r['rollout']['crops'],cpu=r['cpu']['passed'])),flush=True);return r
            except Exception as exc:
                event=dict(audit_id=p['audit_id'],role=role,error_type=type(exc).__name__);errors.append(event)
                write_json(out/'task_errors'/(p['audit_id']+'.json'),event);print(json.dumps(event),flush=True)
                ledger.stop(apis[role].model,'Episode exception; keep unresolved attempts/checkpoint, no auto retry');return None
        def smoke(role):
            if not queues[role]:return
            r=one(role,queues[role].popleft())
            if r is None or r['rollout']['termination'] in ('editor_transport_error','editor_provider_finish_error',
                'editor_schema_or_geometry_rejected','api_error','provider_finish_error','protocol_violation'):
                ledger.stop(apis[role].model,'First actual task transport/identity/read check failed')
        def worker(role):
            while True:
                try:apis[role].check()
                except BudgetStop:return
                with qlock:
                    if not queues[role]:return
                    p=queues[role].popleft()
                one(role,p)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(smoke,apis))
            with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
                fs=[pool.submit(worker,r) for r,s in cfg['models'].items() for _ in range(s['workers'])]
                for f in fs:f.result()
            s=status(out)
            write_json(out/'RUN_STATE.json',dict(state='generation_complete_pending_review' if s['saved']==cfg['tasks'] else 'stopped_incomplete',
                       finished_at=time.time(),errors=errors))
        finally:done.set();watcher.join(timeout=2);status(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','run','status'])
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out.resolve()
    if args.action=='prepare':print(json.dumps(prepare(out),ensure_ascii=False))
    elif args.action=='verify':cfg,items=verify(out,True);print(json.dumps(dict(valid=True,tasks=len(items))))
    elif args.action=='run':run(out)
    else:print(json.dumps(status(out),ensure_ascii=False))
