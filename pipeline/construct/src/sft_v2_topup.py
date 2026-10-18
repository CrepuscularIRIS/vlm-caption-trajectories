"""Bounded V2 repair generation; preserves the 3953-row accepted baseline.

No process judge, promotion, GPU, account switching, or retries of issued requests.
Gemini uses AGY CLI, not an API. CLI invocations and API requests are separate units.
"""
import argparse
import base64
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import requests
from dotenv import dotenv_values

import caption_generation_v2 as base
from agy_reader import AGY, parse_agy
from probe_api import digest
from probe_replica_api import ReplicaAPI
from run_api_t0 import RateLimiter
from tiered_transport import BudgetStop, Ledger, ROOT, write_json

REPORT = ROOT / 'reports/sft_v2_merged_20261008'
SOURCE = REPORT / 'RESULTS.jsonl'
BAD_IDS = 'F00077 F00503 F00660 F02404 F04979 I00055'.split()
UNRESOLVED_IDS = 'F00521 F00703 F00925 F01031 F01185 F01647 F00119 F00299'.split()


def prepare(out):
    if out.exists():
        raise ValueError('Fresh directory required')
    rows = base.read_rows(SOURCE)
    core = [r for r in rows if r['disposition'] == 'core_ready']
    if len(core) != 3953:
        raise ValueError('Recalculate the acceptance gap against the changed baseline')
    core_keys = {r['canonical_sample_key'] for r in core}
    core_groups = {g for r in core for g in r['canonical_groups']}
    corpus_path = ROOT / 'obs/benchmarks/corpus_v2_mmecomplete_20261003/items.jsonl'
    corpus = base.read_rows(corpus_path)
    by_identity = defaultdict(list)
    for c in corpus:
        by_identity[c['image_sha256'], c['question']].append(c)
    demo_paths = [ROOT / 'obs/runs/caption_generation_v2_20261008_r2/examples' / f'demo{i}.json' for i in (1, 2)]
    demo_groups = {json.loads(p.read_text())['group_id'] for p in demo_paths}
    selected, used_groups = [], core_groups | demo_groups
    by_id = {r['audit_id']: r for r in rows}
    excluded = []

    def add(row, cohort):
        if row['canonical_sample_key'] in core_keys or len(row['canonical_groups']) != 1:
            return False
        if set(row['canonical_groups']) & used_groups:
            return False
        if base.sha(row['result_path']) != row['result_sha256']:
            raise ValueError('Old result hash changed')
        old = json.loads(Path(row['result_path']).read_text())['item']
        matches = by_identity.get((row['source_image_sha256'], old['question']), [])
        if not matches or any(c['split'] != 'train' for c in matches):
            return False
        c = matches[0]
        if c['group_id'] != row['canonical_groups'][0]:
            return False
        it = dict(c, actual_sample_key=row['canonical_sample_key'], canonical_group_id=c['group_id'])
        selected.append((row, it, cohort))
        used_groups.add(c['group_id'])
        return True

    for key in BAD_IDS + UNRESOLVED_IDS:
        if not add(by_id[key], 'process_error_repair' if key in BAD_IDS else 'uncertainty_attempt'):
            raise ValueError('Selected diagnostic repair no longer eligible: ' + key)
    weak = [r for r in rows if r['verdict'] == 'WEAK' and r['crops'] > 0
            and r['review']['answer_assessment'] == 'supported'
            and r['review']['binding_assessment'] == 'supported'
            and not r['technical_errors']]
    # Stratify lightly by source, from eligible inventory, not old training quotas.
    for multi, n in [(True, 104), (False, 42)]:
        pools = defaultdict(list)
        for r in weak:
            if (r['crops'] >= 2) == multi and not (set(r['canonical_groups']) & used_groups):
                pools[r['bench']].append(r)
        for pool in pools.values():
            pool.sort(key=lambda r: digest(['topup-v1', r['canonical_sample_key']]))
        weights = {k: math.sqrt(len(v)) for k, v in pools.items()}
        counts = Counter()
        added = 0
        while added < n:
            available = [b for b in pools if pools[b]]
            if not available:
                raise ValueError('Insufficient eligible distinct groups')
            bench = min(available, key=lambda b: ((counts[b] + 1) / weights[b], b))
            r = pools[bench].pop(0)
            if add(r, 'weak_multicrop' if multi else 'weak_onecrop'):
                counts[bench] += 1
                added += 1
    selected_ids = {r['audit_id'] for r, _, _ in selected}
    excluded = [dict(audit_id=r['audit_id'], item_id=r['item_id'], verdict=r['verdict'],
                     reason='not_in_bounded_batch; unresolved/GT ambiguity is not assumed repairable')
                for r in rows if r['verdict'] in ('WEAK', 'BAD', 'UNRESOLVED') and r['audit_id'] not in selected_ids]
    # Updated user allocation: Gemini and K3 only; no Astra dispatch.
    assigned = ['gemini'] * 14 + (['gemini'] * 98 + ['k3'] * 48)
    weak_indices = list(range(14, 160))
    weak_indices.sort(key=lambda i: digest(['allocation', selected[i][0]['canonical_sample_key']]))
    role_by_index = {i: 'gemini' for i in range(14)}
    role_by_index.update(zip(weak_indices, assigned[14:]))
    out.mkdir(parents=True)
    items, private = [], []
    for i, (r, it, cohort) in enumerate(selected):
        it['assigned_role'] = role_by_index[i]
        items.append(it)
        private.append(dict(audit_id=r['audit_id'], item_id=it['item_id'], assigned_role=it['assigned_role'],
                            cohort=cohort, prior_verdict=r['verdict'], prior_crops=r['crops'],
                            previous_result=r['result_path'], previous_result_sha256=r['result_sha256'],
                            rationale_for_selection_only=r['review']['summary_zh'],
                            answerability='prior_supported' if r['verdict'] != 'UNRESOLVED' else 'not_confirmed; independent attempt, no automatic acceptance'))
    # Begin each model with a normal supported repair; no synthetic connectivity call.
    items.sort(key=lambda it: (next(p['prior_verdict'] for p in private if p['item_id'] == it['item_id']) != 'WEAK', digest(it['item_id'])))
    (out / 'items.jsonl').write_text(''.join(json.dumps(i, ensure_ascii=False) + '\n' for i in items))
    write_json(out / 'SELECTION_PRIVATE.json', private)
    write_json(out / 'NOT_SELECTED.json', excluded)
    examples = []
    for p in demo_paths:
        dest = out / 'examples' / p.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, dest)
        examples.append(str(dest))
    extra = [Path(__file__), ROOT / 'obs/src/agy_reader.py', SOURCE, REPORT / 'CORE_READY.jsonl', corpus_path,
             out / 'items.jsonl', out / 'SELECTION_PRIVATE.json', *map(Path, examples)]
    frozen = {str(ROOT / 'obs/src' / p): base.sha(ROOT / 'obs/src' / p) for p in base.CODE_FILES}
    frozen.update({str(p): base.sha(p) for p in extra})
    cfg = dict(version='sft-v2-topup-20261008', created_at=time.time(), deadline=time.time()+8*3600,
               authorization='2026-10-08 user update: Gemini and K3 generation only; Codex performs subsequent review as in V2; monitor and supplement accepted data above 4000',
               tasks=160, workers=4, budget_crops=8, format_retries_per_episode=1, valid_answer_retries=0,
               technical_retries=0, demo_pack_version='train-verified-direct-plus-two-view-v2', examples=examples,
               frozen_files=frozen, corpus=str(corpus_path), baseline_T_Q_G=dict(T=3953, Q=3953, G=3603),
               minimum_new_KEEP_for_over_4000=48, reviewer='Codex_independent_visual_review', no_gpu=True, no_training=True,
               image_policy='unchanged production: original up to 2M pixels; crops from native original; no manufactured blur or forced crop count',
               models={
                   'gemini':dict(model='gemini-3.8-flash-low', effort='low', route='agy', endpoint=AGY,
                                 rpm=12, workers=2, tasks=112, call_cap=1120, max_tokens=8192,
                                 allowed_returned_models=['gemini-3.8-flash-low']),
                   'k3':dict(model='k3-256k', effort='low', route='api', endpoint='http://127.0.0.1:4001',
                             rpm=12, workers=2, tasks=48, call_cap=480, max_tokens=8192, allowed_returned_models=['k3-256k'])},
               budget_units='480 client HTTP requests plus 1120 AGY CLI invocations; AGY internal server requests/retries unobserved and not claimed bounded',
               promotion='none; correct+CPU is only a review candidate. New full-path KEEP with all crops useful and export checks required.',
               no_model_ranking='Different tasks and nonrandom model allocation; this is a repair batch, not a paired benchmark')
    write_json(out / 'config.json', cfg)
    (out / 'CONFIG.sha256').write_text(base.sha(out / 'config.json')+'\n')
    write_json(out / 'SELECTION_STATS.json', dict(tasks=len(items), unique_groups=len({i['group_id'] for i in items}),
               model_tasks=dict(Counter(i['assigned_role'] for i in items)), prior_verdicts=dict(Counter(p['prior_verdict'] for p in private)),
               prior_crops=dict(Counter(p['prior_crops'] for p in private)), cohorts=dict(Counter(p['cohort'] for p in private)),
               benchmarks=dict(Counter(i['bench'] for i in items)), materials=dict(Counter(i['material'] for i in items)),
               baseline_task_overlap=0, baseline_group_overlap=0, demo_group_overlap=0))
    verify(out)
    return cfg


def verify(out):
    cfg, examples, items = base.verify(out)
    if Counter(i['assigned_role'] for i in items) != Counter({r:s['tasks'] for r,s in cfg['models'].items()}):
        raise ValueError('Assignment counts changed')
    core = base.read_rows(REPORT / 'CORE_READY.jsonl')
    if len({i['actual_sample_key'] for i in items}) != len(items) or len({i['group_id'] for i in items}) != len(items):
        raise ValueError('Duplicate task/group')
    if {i['group_id'] for i in items} & {g for r in core for g in r['canonical_groups']}:
        raise ValueError('Overlap with accepted source group')
    return cfg, examples, items


class Transport(ReplicaAPI):
    def __init__(self, out, role, cfg, ledger):
        self.batch, self.role, self.cfg, self.ledger = out, role, cfg, ledger
        self.spec = cfg['models'][role]
        self.model, self.out = self.spec['model'], out / role
        self.max_tokens = self.spec['max_tokens']
        self.lock = threading.Lock()
        self.limiter = RateLimiter(self.spec['rpm'])
        self.consecutive_failures = 0
        for d in ('calls','media','locks','attempts','raw'):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        if self.spec['route'] == 'api':
            ep = base.routing.tm.MODELS[self.model]
            if ep.base != self.spec['endpoint']:
                raise ValueError('Endpoint changed')
            self.ep, self.key = ep, dotenv_values(ep.key_file).get(ep.key_var)
            if not self.key:
                raise ValueError('Missing credential')

    def check(self):
        if base.stopped(self.batch, self.cfg['deadline']):
            raise BudgetStop('Paused/stopped/deadline')
        if (self.ledger.root / ('STOP_'+self.model)).exists():
            raise BudgetStop('Persistent model stop')
        for path, h in self.cfg['frozen_files'].items():
            if base.sha(path) != h:
                write_json(self.batch / 'PAUSE', dict(reason='frozen_source_changed', path=path, time=time.time()))
                raise BudgetStop('Source changed')
        for p in self.cfg.get('examples', []):
            for v in json.loads(Path(p).read_text())['views']:
                if base.sha(v['image_path']) != v['sha256']:
                    write_json(self.batch / 'PAUSE', dict(reason='demo_image_changed', time=time.time()))
                    raise BudgetStop('Demo image changed')

    def call(self, label, msgs, max_tokens=None, *, replicate_id, temperature=.7, top_p=.9):
        safe, images = self.safe_messages(msgs)
        signature = dict(model=self.model, messages=safe, replicate_id=replicate_id, effort=self.spec['effort'], route=self.spec['route'])
        key = digest(signature)
        dest = self.out / 'calls' / (key+'.json')
        with (self.out / 'locks' / (key+'.lock')).open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            if dest.exists():
                return json.loads(dest.read_text())  # Failures are immutable too.
            self.check()
            self.limiter.wait()
            self.check()
            self.ledger.reserve(self.model, key, label, request_limit=1)
            write_json(self.out / 'attempts' / (key+'.json'), dict(state='dispatch_reserved', time=time.time(), request=signature))
            start = time.monotonic()
            try:
                r = self.api_call(msgs, max_tokens, temperature, top_p) if self.spec['route']=='api' else self.agy_call(msgs, key)
            except Exception as exc:
                r = dict(content='', finish_reason='error', returned_model=None, usage={}, errors=[dict(error_type=type(exc).__name__)])
            r.setdefault('errors', [])
            r.setdefault('reasoning_content', '')
            if r.get('returned_model') not in self.spec['allowed_returned_models'] and r['finish_reason'] != 'error':
                r['finish_reason'] = 'error'
                r['errors'].append(dict(error_type='model_identity_mismatch'))
                self.ledger.stop(self.model, 'model_identity_mismatch')
            with self.lock:
                self.consecutive_failures = self.consecutive_failures+1 if r['finish_reason']=='error' else 0
                if self.consecutive_failures >= 3:
                    self.ledger.stop(self.model, 'three_consecutive_transport_failures')
            r.update(label=label, request_sha256=key, request=signature, images=images, route=self.spec['route'],
                     elapsed_s=round(time.monotonic()-start,3), config_sha256=base.sha(self.batch/'config.json'))
            write_json(dest, r)
            write_json(self.out / 'attempts' / (key+'.json'), dict(state='settled', receipt=str(dest), receipt_sha256=base.sha(dest), time=time.time()))
            return r

    def api_call(self, msgs, max_tokens, temperature, top_p):
        ep = self.ep
        payload = dict(model=self.model, messages=msgs, reasoning_effort=self.spec['effort'],
                       temperature=ep.fixed_temperature if ep.fixed_temperature is not None else temperature,
                       top_p=ep.fixed_top_p if ep.fixed_top_p is not None else top_p, max_tokens=max_tokens or self.max_tokens)
        response = requests.post(ep.base+'/v1/chat/completions', json=payload,
                                 headers={'Authorization':'Bearer '+self.key}, timeout=(20,300))
        if response.status_code != 200:
            if response.status_code in (401,402,403,429):
                self.ledger.stop(self.model, 'HTTP '+str(response.status_code))
            return dict(content='',finish_reason='error',returned_model=None,usage={},errors=[dict(status=response.status_code)])
        data=response.json(); choice=data['choices'][0]; message=choice['message']
        return dict(content=message.get('content') or '', reasoning_content=message.get('reasoning_content') or '',
                    finish_reason=choice.get('finish_reason'),returned_model=data.get('model'),usage=data.get('usage',{}),response_id=data.get('id'))

    def agy_call(self, msgs, key):
        wd = Path('/tmp') / ('icml_v2_topup_'+key)
        wd.mkdir(exist_ok=False)
        parts, staged = [], []
        for msg in msgs:
            parts.append('\nMESSAGE ROLE: '+msg['role'])
            content=msg['content']
            for b in ([dict(type='text',text=content)] if isinstance(content,str) else content):
                if b['type']=='text':
                    parts.append(b['text'])
                elif b['type']=='image_url':
                    raw=base64.b64decode(b['image_url']['url'].split(',',1)[1])
                    p=wd / f'image_{len(staged):02d}.jpg'; p.write_bytes(raw); staged.append(p)
                    parts.append('IMAGE ATTACHMENT: '+str(p))
                else:
                    raise ValueError('Unsupported AGY message block')
        prompt = ('Continue the target assistant transcript below for exactly ONE target step, using its Caption/action protocol. '
                  'Open every IMAGE ATTACHMENT with view_file. Use only view_file and finish. Never use commands, network, other files, '
                  'or external knowledge as claimed visual evidence. The demo images are separate tasks. '
                  'If you choose a crop, OUTPUT its grounding tag; do not execute a crop yourself. The environment returns it next step.\n'
                  + '\n'.join(parts))
        cmd=[AGY,'-p',prompt,'--model',self.model,'--effort',self.spec['effort'],'--output-format','stream-json',
             '--sandbox','--disable-slash-commands','--new-project','--print-timeout','5m0s',
             '--log-file',str(self.out / 'raw' / (key+'.agy.log'))]
        write_json(self.out/'raw'/(key+'.request.json'),dict(prompt=prompt,command=cmd[:1]+['-p','<prompt>']+cmd[3:],cwd=str(wd)))
        p=subprocess.Popen(cmd,cwd=wd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           text=True,start_new_session=True)
        try:
            stdout,stderr=p.communicate(timeout=330);code=p.returncode
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(p.pid,signal.SIGTERM)
            try: stdout,stderr=p.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);stdout,stderr=p.communicate()
            code='timeout'
        (self.out/'raw'/(key+'.stdout.jsonl')).write_text(stdout)
        (self.out/'raw'/(key+'.stderr.txt')).write_text(stderr)
        meta=parse_agy(stdout,staged,self.model)
        valid=meta['valid'] and code==0
        if re.search(r'quota|rate.?limit|unauth|authentication|\b429\b|credits.exhaust',stdout+'\n'+stderr,re.I):
            self.ledger.stop(self.model,'AGY quota/auth/rate marker; inspect raw receipt')
        if meta['returned_models'] and meta['returned_models'] != [self.model]:
            self.ledger.stop(self.model,'AGY model mismatch')
        return dict(content=meta['text'],finish_reason='stop' if valid else 'error',
                    returned_model=meta['returned_models'][0] if meta['returned_models'] else None,
                    usage={},errors=[] if valid else [dict(error_type='agy_invalid',exit=code)],
                    agy_receipt=meta,internal_server_request_count=None,workdir=str(wd))


def status(out):
    cfg=json.loads((out/'config.json').read_text()); by={}
    ledger=Ledger(out/'budget',{s['model']:s['call_cap'] for s in cfg['models'].values()})
    attempts=ledger.rows()
    for role,s in cfg['models'].items():
        rows=[json.loads(p.read_text()) for p in (out/role/'results').glob('*.json')]
        calls=[json.loads(p.read_text()) for p in (out/role/'calls').glob('*.json')]
        issued=[a for a in attempts if a['model']==s['model']]
        receipts={r['request_sha256'] for r in calls}
        by[role]=dict(planned=s['tasks'],saved=len(rows),correct=sum(r['score']['score']==1 for r in rows),
            cpu_correct=sum(r['score']['score']==1 and r['cpu']['passed'] for r in rows),
            crops=dict(Counter(r['rollout']['crops'] for r in rows)),terminations=dict(Counter(r['rollout']['termination'] for r in rows)),
            dispatched=len(issued),settled=len(calls),pending_requests=[a['request_sha256'] for a in issued if a['request_sha256'] not in receipts],
            request_errors=sum(r['finish_reason']=='error' for r in calls),returned_models=dict(Counter(str(r.get('returned_model')) for r in calls)),
            unit='AGY_CLI_invocation' if s['route']=='agy' else 'client_HTTP_request',
            stopped=(out/'budget'/('STOP_'+s['model'])).exists(),
            last_receipt_mtime=max((p.stat().st_mtime for p in (out/role/'calls').glob('*.json')),default=None))
    state=json.loads((out/'RUN_STATE.json').read_text()) if (out/'RUN_STATE.json').exists() else dict(state='prepared')
    data=dict(updated_at=time.time(),run_state=state,planned=cfg['tasks'],saved=sum(s['saved'] for s in by.values()),
              generators=by,new_process_KEEP=0,new_accepted_SFT=0,baseline_accepted_SFT=3953)
    write_json(out/'STATUS.json',data); return data


def run(out):
    cfg,examples,items=verify(out)
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'RUN_STATE.json').exists():
            raise ValueError('No automatic restart: inspect pending attempts and obtain explicit resume scope')
        ledger=Ledger(out/'budget',{s['model']:s['call_cap'] for s in cfg['models'].values()})
        apis={r:Transport(out,r,cfg,ledger) for r in cfg['models']}
        queues={r:deque(i for i in items if i['assigned_role']==r) for r in apis}
        write_json(out/'RUN_STATE.json',dict(state='running',pid=os.getpid(),started_at=time.time()))
        done=threading.Event(); errors=[]; qlock=threading.Lock()
        def watch():
            while not done.wait(15): status(out)
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        def one(role,it):
            try:
                apis[role].check()
                if base.sha(it['image'])!=it['image_sha256']:
                    write_json(out/'PAUSE',dict(reason='target_image_changed',item_id=it['item_id'],time=time.time()))
                    raise BudgetStop('Target image changed')
                r=base.generate(out,cfg,examples,it,role,apis[role])
                print(json.dumps(dict(item_id=it['item_id'],role=role,termination=r['rollout']['termination'],
                                      crops=r['rollout']['crops'],score=r['score']['score'],cpu=r['cpu']['passed'])),flush=True)
                return r
            except Exception as exc:
                event=dict(item_id=it['item_id'],role=role,error_type=type(exc).__name__)
                errors.append(event);print(json.dumps(event),flush=True)
                return None
        def smoke(role):
            result=one(role,queues[role].popleft())
            if result is None or result['rollout']['termination'] in ('api_error','provider_finish_error'):
                ledger.stop(apis[role].model,'first_in_batch_transport_check_failed')
        def worker(role):
            while True:
                try: apis[role].check()
                except BudgetStop: return
                with qlock:
                    if not queues[role]: return
                    it=queues[role].popleft()
                one(role,it)
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(smoke,apis))
            status(out)
            with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
                futures=[pool.submit(worker,r) for r,s in cfg['models'].items() for _ in range(s['workers'])]
                for f in futures:f.result()
            s=status(out)
            base.handoff(out)
            write_json(out/'RUN_STATE.json',dict(state='complete_pending_process_review' if s['saved']==cfg['tasks'] else 'stopped_incomplete',
                       finished_at=time.time(),saved=s['saved'],errors=errors,review_owner=cfg['reviewer']))
        except Exception as exc:
            write_json(out/'RUN_STATE.json',dict(state='failed',error_type=type(exc).__name__,time=time.time()))
            raise
        finally:
            done.set();watcher.join(timeout=2);status(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','run','status','handoff'])
    p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve()
    if a.action=='prepare': prepare(out);print(json.dumps(status(out),ensure_ascii=False))
    elif a.action=='verify': cfg,_,items=verify(out);print(json.dumps(dict(valid=True,tasks=len(items))))
    elif a.action=='run': run(out)
    elif a.action=='status': print(json.dumps(status(out),ensure_ascii=False))
    else: print(json.dumps(base.handoff(out),ensure_ascii=False))
