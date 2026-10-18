"""Frozen, bounded production unit. Reuses v5.1 crop execution; keeps old runs untouched."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import time
from pathlib import Path

from PIL import Image
import caption_v51 as proto
from corpus_scoring import score
from run_caption_v51 import rollout
from run_api_t0 import RateLimiter
from tiered_transport import BoundedAPI, CCReviewer, Ledger, BudgetStop, ROOT, write_json

PREP = ROOT / 'obs/notes/traj_replica_20260929/tiered_binding_20261002'
GUIDANCE = ('Record what the current evidence shows, which instance it belongs to, and what remains uncertain. '
            'If candidates remain unresolved, inspect a distinguishing feature. Preserve a supported instance-attribute '
            'binding when checking other evidence. Explicitly revise it if new evidence changes it, and reconsider dependent '
            'conclusions. If a crop lacks the needed context, change the region. Answer directly when the original image is '
            'sufficient. Do not invent an error, unnecessary crop, HOLD or REVISE to match a demonstration.')


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    """Freeze independent train groups and all dispatch limits before model calls."""
    if (out / 'config.json').exists():
        raise ValueError('Existing unit; use run/status, never re-prepare')
    pool = lines(ROOT / 'obs/benchmarks/need_pilot_20261002/label_pool.jsonl')
    labels = {}
    for p in (ROOT / 'obs/runs/need_labels_20261002/labels').glob('*.json'):
        d = json.loads(p.read_text())
        if isinstance(d.get('label'), dict): labels[d['item_id']] = d['label']
    excluded_files = []
    for directory in ('smoke_v51_20261001', 'cover_v51_20261001', 'need_pilot_20261002', 'batch1_20261002'):
        excluded_files.extend((ROOT / 'obs/benchmarks' / directory).glob('items*.jsonl'))
    used = {i['group_id'] for p in excluded_files for i in lines(p)}
    examples = [json.loads(p.read_text()) for p in (PREP / 'fewshot').glob('*__*.json')]
    used.update(e['group_id'] for e in examples)
    existing_sha = {i.get('image_sha256') for p in excluded_files for i in lines(p)} - {None}
    queues = {arm: defaultdict(list) for arm in ('binding', 'keep', 'direct')}
    for it in sorted(pool, key=lambda i: hashlib.sha256(('binding_unit1_20261002/' + i['item_id']).encode()).hexdigest()):
        lab = labels.get(it['item_id'])
        if not lab or it['split'] != 'train' or it['group_id'] in used or it.get('image_sha256') in existing_sha:
            continue
        if lab.get('type') == 'two_region_link': arm = 'keep'
        elif lab.get('type') in ('similar_candidates', 'context_recovery') and not lab.get('legible_at_overview'): arm = 'binding'
        elif lab.get('legible_at_overview'): arm = 'direct'
        else: continue
        queues[arm][(it['bench'], it['material'])].append(it)
    selected, counts = [], Counter()
    for arm, target in [('binding', 30), ('keep', 10), ('direct', 10)]:
        n = 0
        while n < target:
            progress = False
            for _, q in sorted(queues[arm].items()):
                while q and (q[0]['group_id'] in used or counts[q[0]['bench']] >= {'mme_realworld': 25, 'mmmu_pro': 10}.get(q[0]['bench'], 50)):
                    q.pop(0)
                if q and n < target:
                    it = q.pop(0)
                    with Image.open(it['image']) as im: im.load()
                    if sha(it['image']) != it['image_sha256']: raise ValueError('Image hash changed')
                    selected.append(dict(it, production_arm=arm))
                    used.add(it['group_id']); counts[it['bench']] += 1; n += 1; progress = True
            if not progress: raise ValueError(f'Insufficient frozen pool for {arm}: {n}/{target}')
    # First integration items exercise three task types; still part of the same 50 roots.
    first = [next(i for i in selected if i['production_arm'] == arm) for arm in ('direct', 'binding', 'keep')]
    selected = first + [i for i in selected if i not in first]
    k3_ids = [i['item_id'] for i in selected[:3]] + [i['item_id'] for i in selected[3:] if i['production_arm'] != 'direct'][:17]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'items.jsonl').write_text(''.join(json.dumps(i, ensure_ascii=False) + '\n' for i in selected))
    # Full real examples, visually checked 2026-10-02. The six-turn count/HOLD example remains unused.
    example_files = ['mme_realworld_lite_10768__k3-low.json', 'worldbench_1844__glm.json']
    for name in example_files:
        e = json.loads((PREP / 'fewshot' / name).read_text())
        if sha(ROOT / e['source_result']) != e['source_sha256']: raise ValueError('Example source drift')
        if any(sha(v['image_path']) != v['sha256'] for v in e['views']): raise ValueError('Example image drift')
        write_json(out / 'examples' / name, e)
    config = dict(version='binding-tiered-v1', status='frozen_authorized', created_at=time.time(),
                  authorization='2026-10-02 user explicitly authorized mass trajectory generation with cc/agy/API routing and prior quota hierarchy',
                  count=50, planned_trajectories=120, arms=dict(Counter(i['production_arm'] for i in selected)),
                  by_bench=dict(counts), items_sha256=sha(out / 'items.jsonl'),
                  source_pool='obs/benchmarks/corpus_v2', split='train',
                  excluded=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in excluded_files],
                  k3_ids=k3_ids, budget_crops=8, format_retries_per_trajectory=1,
                  generators={'sol': dict(model='gpt-6.1-sol', effort='medium', max_tokens=8192),
                              'glm': dict(model='glm-5.3-flash', effort=None, max_tokens=16384),
                              'k3': dict(model='k3-256k', effort='low', max_tokens=8192)},
                  caps={'gpt-6.1-sol': 500, 'glm-5.3-flash': 600, 'k3-256k': 200,
                        'claude-sonnet-5-5': 200, 'gpt-6-astra': 20, 'claude-opus-5-5': 5},
                  expensive_auto_dispatch=False, technical_attempts_per_request=2,
                  same_valid_answer_retries=0, workers=3, shared_api_rpm=30, review_workers=1,
                  reviewer='claude-sonnet-5-5', reviewer_route='bash task-local cc -> official CLI',
                  gemini_route='agy', gemini_scheduled=False,
                  selection='single_teacher_independent_review', manual_keep_required=True,
                  audit_candidates_per_root=1, first_integration_items=3,
                  examples=[dict(file=n, sha256=sha(out / 'examples' / n)) for n in example_files],
                  seed_review=dict(actor='Codex visual review', provenance='existing human KEEP plus image inspection',
                                   pedagogical='direct sufficient overview; inspect discriminating marker/label binding',
                                   no_successful_correction_demo=True),
                  no_gpu=True, no_training=True, old_runs_untouched=True)
    write_json(out / 'config.json', config)
    return config


class TeacherContext:
    """Scaffold stays outside target transcript/state and the exported SFT labels."""
    def __init__(self, api, example, target_id):
        self.api, self.example, self.target_id = api, example, target_id
        self.format_used = False

    def image(self, path): return self.api.image(path)

    def context(self):
        e = self.example
        def rename(s):
            import re
            return re.sub(r'\b(original_image|observation_\d+)\b', r'demo_\1', s)
        blocks = [dict(type='text', text='COMPLETE DEMONSTRATION — a different completed episode. Demo view names are illustrative only; they cannot be used for the target.\nQuestion: ' + e['question'])]
        by_id = {v['view_id']: v for v in e['views']}
        for step in e['steps']:
            before = 'original_image' if step['step'] == 1 else e['steps'][step['step'] - 2]['observation']
            if before:
                blocks.extend([dict(type='text', text='Received [' + rename(before) + ']'), self.api.image(by_id[before]['image_path'])])
            blocks.append(dict(type='text', text='Demonstration assistant:\n' + rename(step['raw_text'])))
        blocks.append(dict(type='text', text='END DEMONSTRATION. Start a NEW target episode at Step 1. Only target original_image is initially available; all demo images are unavailable for target actions.\n' + GUIDANCE))
        return dict(role='user', content=blocks)

    def call(self, label, msgs, *args, **kwargs):
        if '/fmt' in label:
            if self.format_used:
                return dict(content='', finish_reason='error', errors=[{'error_type': 'trajectory_format_retry_cap'}])
            self.format_used = True
        # Acknowledgement closes the example turn; target messages follow unchanged.
        context = [msgs[0], self.context(), dict(role='assistant', content='The demonstration is complete. I will investigate the new target independently.')]
        return self.api.call(label, context + msgs[1:], *args, **kwargs)


def generate(out, config, ledger, limiter, role, item):
    dest = out / role / 'results' / (item['item_id'] + '.json')
    if dest.exists(): return json.loads(dest.read_text())
    model = config['generators'][role]
    api = BoundedAPI(out / role, ledger=ledger, limiter=limiter, **model)
    example_name = config['examples'][0 if item['production_arm'] == 'direct' else 1]['file']
    example = json.loads((out / 'examples' / example_name).read_text())
    if example['group_id'] == item['group_id']: raise ValueError('Example/target group leak')
    teacher = TeacherContext(api, example, item['item_id'])
    odir = out / role / 'obs' / item['item_id']; odir.mkdir(parents=True, exist_ok=True)
    # Explicit allowlist: target gold, explanations and selection labels cannot enter the runner.
    public_item = {k: item[k] for k in ('item_id', 'question', 'image', 'category', 'material') if k in item}
    ro = rollout(teacher, public_item, odir, config['budget_crops'], .7, .9, model['max_tokens'], format_retries=1)
    ro['export_manifest']['teacher_scaffold_removed'] += ['complete_demonstration', 'mentor_guidance']
    ro['export_manifest'].update(example_id=example['example_id'], example_sha256=sha(out / 'examples' / example_name),
                                 target_state_reset=True, demonstration_labels_supervised=False)
    scoring = score(item['answer_spec'], ro['answer']) if ro['termination'] == 'answer' else dict(score=None, method=ro['termination'])
    result = dict(item=item, model=model['model'], effort=model['effort'], role=role, rollout=ro, score=scoring)
    write_json(dest, result)
    print(json.dumps(dict(event='generated', role=role, item_id=item['item_id'], term=ro['termination'],
                          crops=ro['crops'], correct=scoring['score']), ensure_ascii=False), flush=True)
    return result


def status(out):
    cfg = json.loads((out / 'config.json').read_text())
    result = dict(planned_groups=cfg['count'], planned_trajectories=cfg['planned_trajectories'], generators={})
    for role in cfg['generators']:
        rs = [json.loads(p.read_text()) for p in (out / role / 'results').glob('*.json')]
        result['generators'][role] = dict(completed=len(rs), correct=sum(r['score']['score'] == 1 for r in rs),
                                        crops=sum(r['rollout']['crops'] for r in rs),
                                        terminations=dict(Counter(r['rollout']['termination'] for r in rs)))
    audits = [json.loads(p.read_text()) for p in (out / 'audit').glob('*.json')]
    result['audit'] = dict(completed=len(audits), decisions=dict(Counter(a['decision'] for a in audits)))
    result['dispatch'] = dict(Counter(r['model'] for r in Ledger(out / 'budget', cfg['caps']).rows()))
    exported = out / 'selection' / 'EXPORT_MANIFEST.json'
    result['manual_keep'] = json.loads(exported.read_text())['count'] if exported.exists() else 0
    result['exportable'] = result['manual_keep']
    result['updated_at'] = time.time()
    write_json(out / 'STATUS.json', result)
    return result


def run(out, integration_only=False):
    from audit_tiered import audit_item
    config = json.loads((out / 'config.json').read_text())
    if sha(out / 'items.jsonl') != config['items_sha256']: raise ValueError('Frozen item list changed')
    for e in config['examples']:
        if sha(out / 'examples' / e['file']) != e['sha256']: raise ValueError('Frozen example changed')
    items = lines(out / 'items.jsonl')
    ledger, limiter = Ledger(out / 'budget', config['caps']), RateLimiter(config['shared_api_rpm'])
    reviewer = CCReviewer(out / 'reviewer', ledger)
    with (out / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / 'RUN_STATE.json', dict(state='running', pid=__import__('os').getpid(), time=time.time()))
        try:
            for phase, phase_items in [('integration', items[:3]), ('production', items[3:])]:
                if phase == 'production' and integration_only: break
                with ThreadPoolExecutor(max_workers=config['workers']) as executor:
                    jobs = {}
                    for it in phase_items:
                        for role in config['generators']:
                            if role == 'k3' and it['item_id'] not in config['k3_ids']: continue
                            jobs[executor.submit(generate, out, config, ledger, limiter, role, it)] = (role, it['item_id'])
                    for future in as_completed(jobs):
                        future.result(); status(out)
                for it in phase_items:
                    audit_item(out, config, reviewer, it)
                    status(out)
                from select_tiered import build
                build(out)
                if phase == 'integration':
                    receipts = [json.loads(p.read_text()) for p in (out / 'reviewer').glob('*.json') if not p.name.endswith('.request.json')]
                    if not receipts or any(not r['valid'] for r in receipts):
                        raise RuntimeError('CLI integration gate: missing/invalid role receipt; stop before remaining 47 groups')
                    write_json(out / 'INTEGRATION.json', dict(passed=True, items=[it['item_id'] for it in phase_items],
                                                            role_receipts=len(receipts), validation='transport and pipeline only, not quality acceptance'))
            write_json(out / 'RUN_STATE.json', dict(state='integration_complete' if integration_only else 'complete_pending_manual_review', time=time.time()))
        except Exception as exc:
            write_json(out / 'RUN_STATE.json', dict(state='stopped', error_type=type(exc).__name__, detail=str(exc)[:500], time=time.time()))
            raise
        finally:
            status(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'run', 'status'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--integration-only', action='store_true')
    a = p.parse_args(); out = a.out.resolve()
    if a.action == 'prepare': result = prepare(out)
    elif a.action == 'run': result = run(out, a.integration_only)
    else: result = status(out)
    if result is not None: print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__': main()
