"""Funnel v1 generation: one model, many items, v5.1 real-crop rollouts with one real complete demonstration.

Reuses run_caption_v51.rollout (crop environment, format checks) and caption_tiered.TeacherContext
(demo in its own demo_ view namespace, removed from the student export). Resumable: an item whose result
file exists is skipped; API receipts are reused by request hash. Gold never reaches the generator.

Demo choice depends only on material and a salted hash of the item id (never on gold):
academic / visual_puzzle -> direct demo; other materials -> direct or binding demo, 50/50, recorded.
"""
import argparse
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import models_v51 as routing
from caption_tiered import TeacherContext
from corpus_scoring import score
from run_caption_v51 import rollout

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'obs/runs/binding_unit1_20261002/examples'
DEMOS = {'direct': 'mme_realworld_lite_10768__k3-low.json', 'binding': 'worldbench_1844__glm.json'}
DIRECT_ONLY = {'academic', 'visual_puzzle'}
PUBLIC_FIELDS = ('item_id', 'question', 'image', 'category', 'material')
SALT = 'funnel_v1_demo'
logger = logging.getLogger('funnel_gen')


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demo_for(item: dict) -> str:
    if item['material'] in DIRECT_ONLY:
        return 'direct'
    h = int(hashlib.sha256((SALT + item['item_id']).encode()).hexdigest(), 16)
    return 'binding' if h % 2 else 'direct'


STEP_HEAD = re.compile(r'^\s*Step \d+ \|')
BARE_ACTION = re.compile(r'\n\s*(\{[^\n]*"bbox_2d"[^\n]*\}|<answer>.*?</answer>|<grounding>.*?</grounding>)\s*$', re.S)


GROUNDING_JSON = re.compile(r'<grounding>\s*(\{\s*"bbox_2d"\s*:\s*\[[^\]]*\]\s*,\s*"source"\s*:\s*"[^"]*"\s*\})')
FIRST_ACTION = re.compile(r'<grounding>.*?</grounding>|<answer>.*?</answer>', re.S)


def normalize_tags(text: str) -> str | None:
    """Repair the turn's tag structure without touching caption content. None if nothing to repair.

    - untagged 'Step N | WORD ... {bbox json}' (glm-5.3-flash at low effort): wrap in <caption>/<grounding>;
    - text after the first complete action (gpt-6-luna keeps generating: 'UNCLEAR', '(A) 15', meta remarks): cut;
    - unclosed <grounding>{json} followed by junk or nothing: close after the JSON object and cut.
    """
    if '<caption>' not in text:
        if not STEP_HEAD.match(text):
            return None
        m = BARE_ACTION.search(text)
        if not m:
            return None
        action = m.group(1).strip()
        if action.startswith('{'):
            action = '<grounding>' + action + '</grounding>'
        return '<caption>\n' + text[:m.start()].strip() + '\n</caption>\n' + action
    end = text.find('</caption>')
    if end < 0:
        return None
    m = FIRST_ACTION.search(text, end)
    if m:
        return text[:m.end()] if text[m.end():].strip() else None
    g = GROUNDING_JSON.search(text, end)
    return text[:g.end(1)] + '</grounding>' if g else None


def normalize_kind(raw: str, fixed: str) -> str:
    """wrap_untagged | close_grounding | truncate_trailing (text after the first complete action was dropped;
    the dropped text never enters later history, which is built from the repaired turn)."""
    if '<caption>' not in raw:
        return 'wrap_untagged'
    if raw.startswith(fixed):
        return 'truncate_trailing'
    return 'close_grounding'


class Normalizing:
    """Generator proxy: repairs missing caption/action tags, keeps the original text in the receipt."""
    def __init__(self, gen):
        self.gen = gen
        self.normalized = []  # request hashes whose text was re-tagged

    def image(self, path):
        return self.gen.image(path)

    def call(self, *args, **kwargs):
        r = self.gen.call(*args, **kwargs)
        raw = r.get('raw_text') or r.get('content') or ''
        fixed = normalize_tags(raw) if r.get('finish_reason') == 'stop' else None
        if fixed is None:
            return r
        self.normalized.append(dict(request_sha256=r.get('request_sha256'), kind=normalize_kind(raw, fixed)))
        return dict(r, raw_text=fixed, content=fixed, raw_text_original=raw, format_normalized=True)


class Teacher(TeacherContext):
    """TeacherContext without its one-format-retry-per-trajectory cap (it returned a synthetic 'error' that looked
    like an API failure). rollout(format_retries=1) still bounds format retries to one per step.
    attempt > 0 (a redo after a truncated / protocol-violation trajectory) changes the replicate id, so the redo is
    a fresh sample instead of a replay of the cached failed responses."""
    attempt = 0

    def call(self, label, msgs, *args, **kwargs):
        self.format_used = False
        if self.attempt and 'replicate_id' in kwargs:
            kwargs['replicate_id'] = kwargs['replicate_id'] + f'#redo{self.attempt}'
        return super().call(label, msgs, *args, **kwargs)


def read_redo(out: Path) -> dict:
    p = out / 'REDO.json'
    return json.loads(p.read_text()) if p.exists() else {}


class TechnicalFailure(RuntimeError):
    pass


def generate(api, out: Path, item: dict, examples: dict, budget: int, max_tokens: int, effort: str | None) -> dict | None:
    dest = out / 'results' / (item['item_id'] + '.json')
    if dest.exists():
        return None
    kind = demo_for(item)
    example = examples[kind]
    if example['group_id'] == item['group_id']:
        raise ValueError('Example/target group leak: ' + item['item_id'])
    teacher = Teacher(api, example, item['item_id'])
    redo = read_redo(out).get(item['item_id'])
    if redo:
        teacher.attempt = redo['attempt']
        if redo.get('reason') == 'truncated':
            max_tokens = int(max_tokens * 1.5)
    # Each attempt gets its own view directory so a redo never overwrites the images an archived attempt cited.
    odir = out / 'obs' / (item['item_id'] + (f'.redo{teacher.attempt}' if teacher.attempt else ''))
    odir.mkdir(parents=True, exist_ok=True)
    public = {k: item[k] for k in PUBLIC_FIELDS if k in item}
    t0 = time.time()
    gen = Normalizing(teacher)
    ro = rollout(gen, public, odir, budget, .7, .9, max_tokens, format_retries=1)
    ro['format_normalized_requests'] = gen.normalized
    ro['export_manifest']['teacher_scaffold_removed'] += ['complete_demonstration', 'mentor_guidance']
    ro['export_manifest'].update(example_id=example['example_id'], demo_kind=kind,
                                 target_state_reset=True, demonstration_labels_supervised=False)
    if ro['termination'] == 'api_error':  # not frozen: receipts stay, the item is retried on resume
        errs = [s for s in ro['steps'] if s.get('finish_reason') == 'error']
        raise TechnicalFailure(f"api_error at step {len(ro['steps'])}: " + json.dumps(errs[-1].get('raw_text', ''))[:200])
    scoring = (score(item['answer_spec'], ro['answer']) if ro['termination'] == 'answer'
               else dict(score=None, method=ro['termination']))
    result = dict(item=item, model=api.model, effort=effort, demo_kind=kind, rollout=ro, score=scoring,
                  redo_attempt=teacher.attempt, max_tokens=max_tokens,
                  wall_s=round(time.time() - t0, 1))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    tmp.replace(dest)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--rpm', type=int, default=120)
    p.add_argument('--budget', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--effort', default='low', choices=['low', 'medium', 'high'])
    p.add_argument('--materials', default=None, help='comma list; only items of these materials')
    p.add_argument('--ids-file', type=Path, default=None, help='only these item ids (one per line)')
    p.add_argument('--skip-dirs', default=None, help='comma list of other run dirs; items with a result there are skipped')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if a.model == 'gpt-6-luna':
        routing.tm.MODELS.setdefault('gpt-6-luna', routing.tm.CLIPROXY)  # LiteLLM route returns 500 (thinking kwarg)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    examples = {k: json.loads((EXAMPLES / f).read_text()) for k, f in DEMOS.items()}
    items = [json.loads(s) for s in a.pool.read_text().splitlines() if s.strip()][a.start:a.end]
    if a.materials:
        items = [i for i in items if i['material'] in a.materials.split(',')]
    if a.ids_file:  # keep the ids-file order: queues are priority ordered
        order = list(dict.fromkeys(a.ids_file.read_text().split()))
        first = {}
        for i in items:
            first.setdefault(i['item_id'], i)
        items = [first[i] for i in order if i in first]
    if a.skip_dirs:
        skip = {q.stem for d in a.skip_dirs.split(',') for q in (Path(d) / 'results').glob('*.json')}
        items = [i for i in items if i['item_id'] not in skip]
    run = dict(model=a.model, effort=a.effort, materials=a.materials, n_items=len(items), pool=str(a.pool), pool_sha256=sha(a.pool), start=a.start, end=a.end,
               workers=a.workers, rpm=a.rpm, budget=a.budget, max_tokens=a.max_tokens, time=time.time(),
               demos={k: dict(file=f, sha256=sha(EXAMPLES / f)) for k, f in DEMOS.items()},
               code={f: sha(Path(__file__).parent / f) for f in ('funnel_gen.py', 'run_caption_v51.py', 'caption_v51.py',
                                                                 'caption_tiered.py', 'models_v51.py')})
    (out / 'runs').mkdir(exist_ok=True)
    (out / 'runs' / f'{a.start}-{a.end}_{int(run["time"])}.json').write_text(json.dumps(run, indent=1))
    api = routing.make_api(a.model, out, rpm=a.rpm, max_tokens=a.max_tokens, effort=a.effort)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        jobs = {ex.submit(generate, api, out, it, examples, a.budget, a.max_tokens, a.effort): it['item_id'] for it in items}
        for f in as_completed(jobs):
            try:
                r = f.result()
            except Exception as exc:  # technical failure: logged, item left without result for a later resume
                failed += 1
                logger.error('FAIL %s %s: %s', jobs[f], type(exc).__name__, str(exc)[:300])
                continue
            if r:
                done += 1
                logger.info('done %s term=%s crops=%s score=%s wall=%s', jobs[f], r['rollout']['termination'],
                            r['rollout']['crops'], r['score']['score'], r['wall_s'])
    logger.info('finished: generated=%d failed=%d', done, failed)


if __name__ == '__main__':
    main()
