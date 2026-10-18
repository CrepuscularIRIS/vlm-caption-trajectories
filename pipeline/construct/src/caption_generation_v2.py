"""Generation-only, frozen batches. No reviewer, training, automatic promotion or redraw.

Uses the existing v5.1 environment and original-resolution crop execution. A common
pack of real demonstrations replaces the material-dependent single-demo selector.
All HTTP attempts are reserved durably, including technical retries. Only this
batch's PAUSE/STOP files are consulted; old campaigns cannot be resumed here.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

from dotenv import dotenv_values
from PIL import Image
import requests

from caption_tiered import GUIDANCE
from corpus_scoring import score
from funnel_gen import normalize_kind, normalize_tags
import models_v51 as routing
from probe_api import digest
from probe_replica_api import ReplicaAPI
from run_api_t0 import RateLimiter
from run_caption_v51 import rollout
from tiered_transport import BudgetStop, Ledger, ROOT, write_json

VERSION = 'caption-generation-v2-20261008'
PUBLIC_FIELDS = ('item_id', 'question', 'image', 'category', 'material')
CODE_FILES = ('caption_generation_v2.py', 'run_caption_v51.py', 'caption_v51.py',
              'run_minio3_replica.py', 'monitor_step.py', 'funnel_gen.py',
              'caption_tiered.py', 'models_v51.py', 'trajgen_models.py',
              'tiered_transport.py', 'probe_api.py', 'probe_replica_api.py',
              'run_api_t0.py', 'corpus_scoring.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def stopped(out, deadline):
    return (out / 'PAUSE').exists() or (out / 'STOP').exists() or time.time() >= deadline


class DemoTeacher:
    """Independent demo namespaces; no target gold, old path, label or target hint injection."""
    def __init__(self, api, examples):
        self.api, self.examples = api, examples
        self.format_retry_remaining = 1
        self.normalized = []

    def image(self, path):
        return self.api.image(path)

    def context(self):
        blocks = []
        for i, e in enumerate(self.examples, 1):
            def rename(s):
                return re.sub(r'\b(original_image|observation_\d+)\b', rf'demo{i}_\1', s)
            blocks.append(dict(type='text', text='COMPLETE DEMONSTRATION of a separate task.\nQuestion: ' + e['question']))
            by_id = {v['view_id']: v for v in e['views']}
            before = 'original_image'
            for step in e['steps']:
                if before:
                    blocks.extend([dict(type='text', text='Received [' + rename(before) + ']'),
                                   self.api.image(by_id[before]['image_path'])])
                blocks.append(dict(type='text', text='Demonstration assistant:\n' + rename(step['raw_text'])))
                before = step.get('observation')
            blocks.append(dict(type='text', text='END DEMONSTRATION. This episode is complete.'))
        blocks.append(dict(type='text', text='Start a NEW target episode at Step 1. Only target original_image is initially available. '
                           'No demo view or fact can be used as target evidence. The examples do not specify how many crops '
                           'the target needs.\n' + GUIDANCE))
        return dict(role='user', content=blocks)

    def call(self, label, msgs, *args, **kwargs):
        if '/fmt' in label:
            if self.format_retry_remaining <= 0:
                raise RuntimeError('Episode format budget exhausted before dispatch')
            self.format_retry_remaining -= 1
        prefix = [msgs[0], self.context(), dict(role='assistant', content='I will investigate the new target independently.')]
        r = self.api.call(label, prefix + msgs[1:], *args, **kwargs)
        raw = r.get('raw_text') or r.get('content') or ''
        fixed = normalize_tags(raw) if r.get('finish_reason') == 'stop' else None
        if fixed is not None:
            self.normalized.append(dict(request_sha256=r['request_sha256'], kind=normalize_kind(raw, fixed)))
            return dict(r, content=fixed, raw_text=fixed, raw_text_original=raw)
        return r


class GenerationAPI(ReplicaAPI):
    """Explicit route, bounded retries, immutable receipts and persistent quota stop."""
    def __init__(self, out, spec, ledger, batch, deadline):
        if spec['model'].startswith('claude'):
            raise ValueError('Review belongs to the user; no Claude route in this runner')
        ep = routing.tm.MODELS[spec['model']]
        if ep.base != spec['endpoint']:
            raise ValueError('Endpoint drift')
        super().__init__(out, spec['model'], rpm=spec['rpm'], max_tokens=spec['max_tokens'])
        self.base, self.key = ep.base, dotenv_values(ep.key_file).get(ep.key_var)
        if not self.key:
            raise ValueError('Missing credential for ' + spec['model'])
        self.ep, self.spec, self.ledger, self.batch, self.deadline = ep, spec, ledger, batch, deadline

    def call(self, label, msgs, max_tokens=None, *, replicate_id, temperature=.7, top_p=.9):
        safe, images = self.safe_messages(msgs)
        payload = dict(model=self.model, messages=msgs, max_tokens=max_tokens or self.max_tokens,
                       reasoning_effort=self.spec['effort'],
                       temperature=self.ep.fixed_temperature if self.ep.fixed_temperature is not None else temperature,
                       top_p=self.ep.fixed_top_p if self.ep.fixed_top_p is not None else top_p)
        signature = dict(payload, messages=safe, replicate_id=replicate_id)
        key = digest(signature)
        dest = self.out / 'calls' / (key + '.json')
        with (self.out / 'locks' / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if dest.exists():
                return json.loads(dest.read_text())  # failures are also frozen, never implicitly redrawn
            errors, result, start = [], None, time.monotonic()
            for _ in range(2):
                if stopped(self.batch, self.deadline):
                    raise BudgetStop('Batch paused, stopped, or deadline reached')
                self.limiter.wait()
                if stopped(self.batch, self.deadline):
                    raise BudgetStop('Batch stopped before dispatch')
                dispatch = self.ledger.reserve(self.model, key, label, request_limit=2)
                attempt = dispatch['attempt']
                try:
                    response = requests.post(self.base + '/v1/chat/completions', json=payload,
                                             headers={'Authorization': 'Bearer ' + self.key}, timeout=(20, 300))
                    if response.status_code != 200:
                        # Provider bodies can echo payloads or credentials; keep status only.
                        errors.append(dict(status=response.status_code, attempt=attempt))
                        if response.status_code in (401, 402, 403, 429):
                            self.ledger.stop(self.model, 'HTTP ' + str(response.status_code))
                        if response.status_code < 500:
                            break
                    else:
                        data = response.json(); choice = data['choices'][0]; message = choice['message']
                        result = dict(content=message.get('content') or '', reasoning_content=message.get('reasoning_content') or '',
                                      finish_reason=choice.get('finish_reason'), returned_model=data.get('model'),
                                      usage=data.get('usage', {}), response_id=data.get('id'))
                        if result['returned_model'] not in self.spec['allowed_returned_models']:
                            result['finish_reason'] = 'error'
                            errors.append(dict(error_type='returned_model_mismatch', attempt=attempt))
                            self.ledger.stop(self.model, 'Returned model outside frozen allowlist')
                        break
                except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                    errors.append(dict(error_type=type(exc).__name__, attempt=attempt))
                if attempt >= 2:
                    break
                time.sleep(2)
            result = result or dict(content='', reasoning_content='', finish_reason='error', returned_model=None, usage={})
            result.update(label=label, request=signature, request_sha256=key, images=images,
                          endpoint=self.base, alias_identity_verified=self.spec['alias_identity_verified'],
                          errors=errors, elapsed_s=round(time.monotonic() - start, 3))
            write_json(dest, result)
            return result


def verify(out):
    if sha(out / 'config.json') != (out / 'CONFIG.sha256').read_text().strip():
        raise ValueError('Frozen config changed')
    cfg = json.loads((out / 'config.json').read_text())
    for p, expected in cfg['frozen_files'].items():
        if sha(p) != expected:
            raise ValueError('Frozen input/code changed: ' + p)
    examples = [json.loads(Path(p).read_text()) for p in cfg['examples']]
    items = read_rows(out / 'items.jsonl')
    corpus = read_rows(cfg['corpus'])
    by_hash, by_id, splits_by_group = {}, {}, {}
    for row in corpus:
        by_hash.setdefault(row['image_sha256'], []).append(row)
        by_id.setdefault(row['item_id'], []).append(row)
        splits_by_group.setdefault(row['group_id'], set()).add(row['split'])
    demo_groups = set()
    for e in examples:
        matches = by_hash.get(e['source_image_sha256'], [])
        # Legacy Lite encodings can have a different hash but the same full-dataset item.
        if not matches and e.get('item_id', '').startswith('mme_realworld_lite_'):
            matches = by_id.get(e['item_id'].replace('mme_realworld_lite_', 'mme_realworld_'), [])
        matches = [r for r in matches if r['question'] == e['question']]
        if not matches or any(r['split'] != 'train' or splits_by_group[r['group_id']] != {'train'} for r in matches):
            raise ValueError('Demo is not exclusively in current train corpus')
        demo_groups.update(r['group_id'] for r in matches)
    ids, keys, groups = set(), set(), set()
    for it in items:
        if it['split'] != 'train' or it['item_id'] in ids or it['actual_sample_key'] in keys:
            raise ValueError('Invalid split or duplicate sample')
        if sha(it['image']) != it['image_sha256']:
            raise ValueError('Target image changed')
        current = [r for r in by_hash.get(it['image_sha256'], []) if r['question'] == it['question']]
        if not current or any(r['group_id'] != it['canonical_group_id'] or r['split'] != 'train'
                              or splits_by_group[r['group_id']] != {'train'} for r in current):
            raise ValueError('Target identity or split does not match current corpus')
        if it['canonical_group_id'] in demo_groups or any(it['image_sha256'] == e['source_image_sha256'] for e in examples):
            raise ValueError('Demo/target group or image overlap')
        ids.add(it['item_id']); keys.add(it['actual_sample_key']); groups.add(it['canonical_group_id'])
    for e in examples:
        if sha(e['source_result']) != e['source_sha256']:
            raise ValueError('Demo source result changed')
        for v in e['views']:
            if sha(v['image_path']) != v['sha256']:
                raise ValueError('Demo view changed')
    if len(items) != cfg['tasks']:
        raise ValueError('Task count differs from frozen config')
    return cfg, examples, items


def cpu_check(ro):
    reasons = []
    if ro['termination'] != 'answer':
        reasons.append(ro['termination'])
    for step in ro['steps']:
        reasons.extend(step.get('violations', []))
        reasons.extend(f for f in step.get('flags', []) if f in ('coordinate_talk', 'coordinate_unresolved'))
    return dict(passed=not reasons, reasons=sorted(set(reasons)))


def generate(out, cfg, examples, it, role, api):
    dest = out / role / 'results' / (it['item_id'] + '.json')
    if dest.exists():
        return json.loads(dest.read_text())
    if stopped(out, cfg['deadline']):
        raise BudgetStop('Not starting more episodes')
    odir = out / role / 'obs' / it['item_id']; odir.mkdir(parents=True, exist_ok=True)
    teacher = DemoTeacher(api, examples)
    # GT and old review are deliberately excluded from the generator's item object.
    public = {k: it[k] for k in PUBLIC_FIELDS if k in it}
    start = time.time()
    ro = rollout(teacher, public, odir, cfg['budget_crops'], .7, .9, cfg['models'][role]['max_tokens'], format_retries=1)
    ro['format_normalized_requests'] = teacher.normalized
    ro['export_manifest']['teacher_scaffold_removed'] += ['complete_demonstrations', 'mentor_guidance']
    ro['export_manifest'].update(demo_pack=cfg['demo_pack_version'], example_ids=[e['example_id'] for e in examples],
                                 target_state_reset=True, demonstration_labels_supervised=False)
    r = dict(item=it, model=api.model, effort=cfg['models'][role]['effort'], rollout=ro,
             score=score(it['answer_spec'], ro['answer']) if ro['termination'] == 'answer' else dict(score=None, method=ro['termination']),
             cpu=cpu_check(ro), wall_s=round(time.time() - start, 2), config_sha256=sha(out / 'config.json'),
             version=VERSION, review_state='NOT_REVIEWED', format_retries_used=1 - teacher.format_retry_remaining)
    write_json(dest, r)
    return r


def status(out):
    cfg = json.loads((out / 'config.json').read_text())
    by_role = {}
    for role in cfg['models']:
        rows = [json.loads(p.read_text()) for p in (out / role / 'results').glob('*.json')]
        calls = [json.loads(p.read_text()) for p in (out / role / 'calls').glob('*.json')]
        by_role[role] = dict(planned=cfg['tasks'], saved=len(rows),
                            correct=sum(r['score']['score'] == 1 for r in rows),
                            cpu_correct=sum(r['score']['score'] == 1 and r['cpu']['passed'] for r in rows),
                            crops=dict(Counter(str(r['rollout']['crops']) for r in rows)),
                            terminations=dict(Counter(r['rollout']['termination'] for r in rows)),
                            returned_models=dict(Counter(str(r.get('returned_model')) for r in calls)),
                            requests=len(calls), request_errors=sum(r['finish_reason'] == 'error' for r in calls),
                            completion_tokens=sum(r.get('usage', {}).get('completion_tokens', 0) for r in calls),
                            total_tokens=sum(r.get('usage', {}).get('total_tokens', 0) for r in calls),
                            last_receipt_mtime=max([p.stat().st_mtime for p in (out / role / 'calls').glob('*.json')], default=None))
    ledger = Ledger(out / 'budget', {s['model']: s['http_cap'] for s in cfg['models'].values()})
    state = json.loads((out / 'RUN_STATE.json').read_text()) if (out / 'RUN_STATE.json').exists() else dict(state='prepared')
    s = dict(updated_at=time.time(), state=state, tasks=cfg['tasks'], planned_trajectories=cfg['tasks'] * len(cfg['models']),
             generators=by_role, actual_http_attempts=dict(Counter(r['model'] for r in ledger.rows())),
             process_reviewed=0, accepted_sft=0, review_owner='user_CC_workflow')
    write_json(out / 'STATUS.json', s)
    return s


def handoff(out):
    """GT-free stage 1 has only obs0; stage 2 contains the actual full path. No judge call."""
    cfg = json.loads((out / 'config.json').read_text()); private, index = [], []
    for role in cfg['models']:
        for p in sorted((out / role / 'results').glob('*.json')):
            r = json.loads(p.read_text()); ro, it = r['rollout'], r['item']
            key = hashlib.sha256((role + '/' + it['actual_sample_key']).encode()).hexdigest()[:16]
            d = out / 'review_handoff' / key
            views = []
            for n, v in enumerate(ro['views']):
                # Neutral copies avoid revealing the generator through directory names.
                view_path = d / 'views' / f'obs{n}.jpg'
                view_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(v['path'], view_path)
                if sha(view_path) != v['sha256']:
                    raise ValueError('Handoff view hash mismatch')
                views.append(dict(view_id=v['view_id'], image_path=str(view_path), sha256=v['sha256'],
                                  shown_px=v['shown_px'], parent=v['parent'], fov_original=v['fov_original']))
            write_json(d / '01_blind_question.json', dict(review_id=key, question=it['question'], views=views[:1],
                       instruction='Read only this stage first. Record supported answer or uncertainty before opening stage 2; do not claim an independent blind answer after seeing the trajectory.'))
            write_json(d / '02_trajectory.json', dict(review_id=key, question=it['question'], views=views,
                       steps=[dict(step=s['step'], available_before=s['available_before'], raw_text=s['raw_text'],
                                   observation=s.get('observation')) for s in ro['steps']], termination=ro['termination'],
                       instruction='Use only the listed actual views. Each claim is checked against views already available at that step. KEEP requires reliable key facts and binding, not crop count. Direct answers are allowed.'))
            private.append(dict(review_id=key, role=role, item_id=it['item_id'], result=str(p), result_sha256=sha(p),
                                answer_spec=it['answer_spec'], score=r['score'], cpu=r['cpu']))
            index.append(dict(review_id=key, blind_question=str(d / '01_blind_question.json'), trajectory=str(d / '02_trajectory.json')))
    write_json(out / 'review_handoff' / 'INDEX.json', index)
    write_json(out / 'SCORING_AND_REVIEW_MAP_PRIVATE.json', private)
    return dict(paths=len(index), review_index=str(out / 'review_handoff' / 'INDEX.json'))


def run(out):
    cfg, examples, items = verify(out)
    with (out / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stopped(out, cfg['deadline']):
            raise BudgetStop('Batch paused, stopped, or expired')
        ledger = Ledger(out / 'budget', {s['model']: s['http_cap'] for s in cfg['models'].values()})
        apis = {r: GenerationAPI(out / r, spec, ledger, out, cfg['deadline']) for r, spec in cfg['models'].items()}
        write_json(out / 'RUN_STATE.json', dict(state='running', pid=os.getpid(), started_at=time.time()))
        stop_monitor = threading.Event()
        def monitor():
            while not stop_monitor.wait(15):
                status(out)
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        failures = []
        try:
            with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
                jobs = {pool.submit(generate, out, cfg, examples, it, role, apis[role]): (it['item_id'], role)
                        for it in items for role in cfg['models']}
                for f in as_completed(jobs):
                    item_id, role = jobs[f]
                    try:
                        r = f.result()
                        event = dict(item_id=item_id, role=role, termination=r['rollout']['termination'],
                                     crops=r['rollout']['crops'], score=r['score']['score'], cpu=r['cpu']['passed'])
                    except Exception as exc:
                        event = dict(item_id=item_id, role=role, error_type=type(exc).__name__)
                        failures.append(event)
                    print(json.dumps(event), flush=True)
            saved = sum(len(list((out / role / 'results').glob('*.json'))) for role in cfg['models'])
            write_json(out / 'RUN_STATE.json', dict(state='complete_pending_user_review' if saved == cfg['tasks'] * len(cfg['models']) else 'stopped_incomplete',
                       finished_at=time.time(), failures=failures, saved=saved))
            handoff(out)
        finally:
            stop_monitor.set(); watcher.join(timeout=2)
            status(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['verify', 'run', 'status', 'handoff'])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); out = a.out.resolve()
    if a.action == 'verify':
        cfg, _, items = verify(out)
        print(json.dumps(dict(valid=True, tasks=len(items), planned=len(items) * len(cfg['models']))))
    elif a.action == 'run':
        run(out)
    else:
        print(json.dumps(status(out) if a.action == 'status' else handoff(out), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
