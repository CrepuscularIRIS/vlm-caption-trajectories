"""Mini-o3 App. F cold-start replication: exemplar-prompted VLM + real crop environment.

Per question: (1) direct-answer baseline (no tool), (2) one multi-turn rollout where the
generator emits {"thought","action"} JSON, the environment crops the full-resolution source
image and appends the observation, until final() or the turn limit. Final answers are judged
with the Mini-o3 Table 5 text-only judge. Kept rollouts are converted to the Coldstart format.
"""
import argparse
import fcntl
import glob
import hashlib
import json
import logging
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from minio3_prompts import (TABLE5_JUDGE_SYSTEM, TABLE5_JUDGE_USER, TABLE8_INSTRUCTION, system_prompt,
                            turn_instruction)
from behavior_prompts import system_prompt_v1, turn_instruction_v1
from monitor_step import StepMonitor, expand_min_crop, leak_flag, lenient_parse, revise_message
from probe_api import JSON_INSTR, structured, text_msg
from trajgen_models import JUDGE, make_api

ROOT = Path('${ICML_DIR}')
HUB = '${HF_HOME}/hub/'
MAX_PIXELS = 2_000_000
MIN_SIDE = 448          # tiny crops are upsampled (<=4x) so the API does not reject/over-shrink them
MAX_UPSCALE = 4.0
LOG = logging.getLogger('minio3')

GROUND = re.compile(r'grounding\(\s*\[([^\]]*)\]\s*,\s*(?:source\s*=\s*)?["\'‘’“”]?\s*([A-Za-z0-9_ ]+?)\s*["\'‘’“”]?\s*\)', re.S)
FINAL = re.compile(r'final\(\s*(.*)\s*\)\s*$', re.S)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snap(name):
    return glob.glob(HUB + f'datasets--Mini-o3--{name}/snapshots/*/')[0]


def load_pool(n, seed):
    """VisualProbe train questions whose image is not byte-identical to any VisualProbe test image."""
    test_sha = set()
    for lv in ('Easy', 'Medium', 'Hard'):
        s = snap('VisualProbe_' + lv)
        js = [f for f in glob.glob(s + '**', recursive=True) if f.endswith('.json')][0]
        for x in json.load(open(js)):
            test_sha.add(sha_file(s + x['images'][0].split('/', 1)[1]))
    s = snap('VisualProbe_train')
    train = json.load(open(s + 'train.json'))
    rng = random.Random(seed)
    rng.shuffle(train)
    pool, excluded = [], 0
    for x in train:
        path = s + x['images'][0].split('/', 1)[1]
        h = sha_file(path)
        if h in test_sha:
            excluded += 1
            continue
        pool.append(dict(doc_id=x['doc_id'], question=x['problem'].replace('<image>', '').strip(),
                         gold=x['solution'], image=path, image_sha256=h))
        if len(pool) >= n:
            break
    return pool, excluded


def fit(im):
    """Downscale to <=MAX_PIXELS; upsample small crops to MIN_SIDE (<=MAX_UPSCALE)."""
    w, h = im.size
    if min(w, h) < MIN_SIDE:
        f = min(MIN_SIDE / min(w, h), MAX_UPSCALE)
        w, h = max(28, round(w * f)), max(28, round(h * f))
    nh, nw = smart_resize(h, w, max_pixels=MAX_PIXELS)
    return im.resize((nw, nh), Image.Resampling.BICUBIC) if (nw, nh) != im.size else im


def save_obs(im, dest):
    fit(im).convert('RGB').save(dest, 'JPEG', quality=95)
    return dest


def parse_action(content, lenient=False):
    obj = structured(content)
    if obj is None:
        m = re.search(r'\{.*\}', content, re.S)
        obj = structured(m.group(0)) if m else None
    mode = 'json'
    if (not obj or not isinstance(obj.get('action'), str)) and lenient:
        obj, mode = lenient_parse(content), 'lenient'
    if not obj or not isinstance(obj.get('action'), str):
        return None, 'unparseable_json'
    thought, action = str(obj.get('thought', '')), obj['action'].strip()
    behavior = str(obj.get('behavior', '')).strip().lower() or None
    m = FINAL.match(action)
    if m:
        return dict(thought=thought, action=action, kind='final', parse_mode=mode, behavior=behavior,
                    answer=m.group(1).strip().strip('"\'“”')), None
    m = GROUND.search(action)
    if not m:
        return dict(thought=thought, action=action, kind='unknown', parse_mode=mode), 'unknown_action'
    try:
        box = [float(v) for v in m.group(1).split(',')]
    except ValueError:
        return dict(thought=thought, action=action, kind='grounding', parse_mode=mode), 'bad_bbox'
    src = m.group(2).strip().lower().replace(' ', '_')
    return dict(thought=thought, action=action, kind='grounding', bbox=box, source=src, parse_mode=mode,
                behavior=behavior), None


def resolve_source(src, n_obs):
    if src in ('original_image', 'original'):
        return 0
    m = re.fullmatch(r'observation_?(\d+)', src)
    if m and 1 <= int(m.group(1)) < n_obs:
        return int(m.group(1))
    return None


def valid_box(b):
    return (len(b) == 4 and all(0 <= v <= 1 for v in b) and b[2] - b[0] >= 0.01 and b[3] - b[1] >= 0.01)


def _generate(api, item, msgs, i, tag, max_tokens, temperature, top_p, lenient):
    r = api.call(f"{item['doc_id']}/t{i}{tag}", msgs, max_tokens, replicate_id=item['doc_id'] + '#r0',
                 temperature=temperature, top_p=top_p)
    step = dict(turn=i, request_sha256=r['request_sha256'], finish_reason=r['finish_reason'],
                returned_model=r['returned_model'], usage=r['usage'], content=r['content'],
                reasoning_chars=len(r.get('reasoning_content') or ''))
    if r['finish_reason'] == 'error':
        return step, None, 'api_error'
    if r['finish_reason'] != 'stop':
        return step, None, 'truncated'
    act, err = parse_action(r['content'], lenient)
    return step, act, err


def rollout(api, item, odir, max_actions, temperature, top_p, max_tokens, opts=None):
    """opts: min_crop_px (0=off), lenient (bool), monitor (StepMonitor|None), max_revisions."""
    opts = opts or {}
    full = Image.open(item['image']).convert('RGB')
    W, H = full.size
    boxes = [(0.0, 0.0, float(W), float(H))]            # observation k -> absolute box in full-res coords
    paths = [save_obs(full, odir / 'obs0.jpg')]
    sys_fn, turn_fn = opts.get('prompts') or (system_prompt, turn_instruction)
    msgs = [text_msg('system', sys_fn()),
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f"Question: {item['question']}\nOriginal Image:"},
                api.image(str(paths[0])), {'type': 'text', 'text': turn_fn(0)}]}]
    steps, term, monitor = [], None, opts.get('monitor')
    for i in range(max_actions + 1):
        step, act, err = _generate(api, item, msgs, i, '', max_tokens, temperature, top_p, opts.get('lenient'))
        if monitor is not None and not err:
            reviews, rejected = [], []
            latest = paths[-1] if len(paths) > 1 else None
            budget = opts.get('max_revisions', 2)
            if opts.get('trajectory_revision_cap') is not None:
                budget = min(budget, max(0, opts['trajectory_revision_cap'] - sum(s.get('revisions', 0) for s in steps)))
            if act['kind'] == 'final' and opts.get('final_flag_only', True):
                budget = 0   # A/B 2026-09-30: forcing a redo of final answers flipped right->wrong 9x vs wrong->right 7x
            for attempt in range(budget + 1):
                rv = monitor.review(item['doc_id'], item['question'], paths[0], latest, i, attempt,
                                    act['thought'], act['action'])
                reviews.append(rv)
                rv['hint_leaks_gold'] = leak_flag(rv['hint'] + ' ' + ' '.join(map(str, rv['problems'])), str(item['gold']))
                if rv['verdict'] != 'revise' or attempt == budget:
                    break
                # Redo this step with the hint; the rejected attempt and the note are NOT kept in context.
                rejected.append(step['content'])
                redo = msgs + [{'role': 'assistant', 'content': step['content']},
                               {'role': 'user', 'content': [{'type': 'text', 'text':
                                   revise_message(i, rv) + '\n' + turn_fn(i)}]}]
                step2, act2, err2 = _generate(api, item, redo, i, f'_rev{attempt + 1}', max_tokens,
                                              temperature, top_p, opts.get('lenient'))
                if err2:
                    rv['redo_error'] = err2
                    break
                step, act = step2, act2
            step.update(monitor_reviews=reviews, rejected_contents=rejected, revisions=len(rejected),
                        monitor_unresolved=reviews[-1]['verdict'] == 'revise')
        steps.append(step)
        step.update(act or {})
        if err:
            term = err; break
        if act['kind'] == 'final':
            term = 'final'; break
        if i == max_actions:
            term = 'turn_limit'; break
        k = resolve_source(act['source'], len(boxes))
        if k is None or not valid_box(act['bbox']):
            term = 'invalid_grounding'; step['invalid'] = True; break
        x0, y0, x1, y1 = boxes[k]
        b = act['bbox']
        ab = (x0 + b[0] * (x1 - x0), y0 + b[1] * (y1 - y0), x0 + b[2] * (x1 - x0), y0 + b[3] * (y1 - y0))
        if opts.get('min_crop_px'):
            ab, grew = expand_min_crop(ab, W, H, opts['min_crop_px'])
            step['crop_expanded'] = grew
        crop = full.crop(tuple(round(v) for v in ab))
        n = len(boxes)
        boxes.append(ab)
        paths.append(save_obs(crop, odir / f'obs{n}.jpg'))
        step.update(source_index=k, abs_box=[round(v, 1) for v in ab], observation=n,
                    crop_px=list(crop.size), obs_sha256=sha_file(paths[-1]))
        msgs += [{'role': 'assistant', 'content': step['content']},
                 {'role': 'user', 'content': [
                     {'type': 'text', 'text': f'Observation {n}:'}, api.image(str(paths[-1])),
                     {'type': 'text', 'text': turn_fn(n)}]}]
    return dict(steps=steps, termination=term, source_size=[W, H],
                answer=steps[-1].get('answer') if term == 'final' else None,
                n_grounding=sum(s.get('kind') == 'grounding' and 'observation' in s for s in steps),
                obs_paths=[str(p.relative_to(ROOT)) for p in paths])


def judge(api, item, pred, tag):
    if pred is None:
        return None
    msgs = [text_msg('system', TABLE5_JUDGE_SYSTEM),
            text_msg('user', TABLE5_JUDGE_USER.format(question=item['question'], gold=item['gold'], pred=pred))]
    r = api.call(f"{item['doc_id']}/judge_{tag}", msgs, 2048, replicate_id='judge', temperature=0.)
    m = re.search(r'Score:\s*([01])', r['content'])
    return dict(score=int(m.group(1)) if m else None, request_sha256=r['request_sha256'])


def direct(api, item, odir):
    img = api.image(str(odir / 'obs0.jpg'))
    msgs = [{'role': 'user', 'content': [img, {'type': 'text', 'text': item['question'] + '\n' + JSON_INSTR}]}]
    r = api.call(f"{item['doc_id']}/direct", msgs, 4096, replicate_id=item['doc_id'] + '#direct', temperature=0.)
    obj = structured(r['content']) if r['finish_reason'] == 'stop' else None
    ans = str(obj['answer']).strip() if obj and 'answer' in obj else None
    return dict(answer=ans, finish_reason=r['finish_reason'], request_sha256=r['request_sha256'])


def to_coldstart(item, ro):
    """Convert a kept rollout to the released Mini-o3 Coldstart training format."""
    conv = [{'from': 'human', 'value': '<image>\n' + item['question']}]
    for s in ro['steps']:
        if s.get('kind') == 'final':
            conv.append({'from': 'gpt', 'value': f"<think> {s['thought']} </think><answer> {s['answer']} </answer>"})
            break
        src = 'original_image' if s['source_index'] == 0 else f"observation_{s['source_index']}"
        g = json.dumps({'bbox_2d': s['bbox'], 'source': src})
        conv.append({'from': 'gpt', 'value': f"<think> {s['thought']} </think><grounding>{g}</grounding>"})
        conv.append({'from': 'human', 'value': f"After the above Action {s['turn']}, here is the the zoom-in image "
                     f"(Observation {s['observation']}):\n<image>.\nContinue your reasoning process inside <think> and </think>. "
                     "If needed, you can continue to zoom in on the original image or any of the observations, by outputing "
                     "<grounding> and </grounding> as before. If the final answer is confirmed, put your final answer inside "
                     "<answer> and </answer>."})
    return dict(doc_id=item['doc_id'], conversations=conv, images=ro['obs_paths'])


def run_one(api, judge_api, item, out, args):
    dest = out / 'results' / f"{item['doc_id']}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    odir = out / 'obs' / item['doc_id']
    odir.mkdir(parents=True, exist_ok=True)
    ro = rollout(api, item, odir, args.max_actions, args.temperature, args.top_p, args.max_tokens, args.opts)
    d = direct(api, item, odir)
    res = dict(item, rollout=ro, direct=d,
               judge_rollout=judge(judge_api, item, ro['answer'], 'rollout'),
               judge_direct=judge(judge_api, item, d['answer'], 'direct'))
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    tmp.replace(dest)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default='glm-5.3-flash')
    p.add_argument('--max-actions', type=int, default=8)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--rpm', type=int, default=40)
    p.add_argument('--items', type=Path, default=None, help='fixed items.jsonl (overrides --n/--seed pool)')
    p.add_argument('--effort', default='medium', help="reasoning_effort for the generator; 'none' to omit")
    p.add_argument('--judge-model', default=JUDGE)
    p.add_argument('--min-crop-px', type=int, default=0, help='enlarge crops below this source-pixel size (0=off)')
    p.add_argument('--lenient-parse', action='store_true')
    p.add_argument('--monitor', default=None, help='online step monitor model (None=off)')
    p.add_argument('--prompt-set', choices=['minio3', 'behavior_v1'], default='minio3')
    p.add_argument('--max-revisions', type=int, default=2, help='per step')
    p.add_argument('--trajectory-revision-cap', type=int, default=3)
    args = p.parse_args()
    out = args.out
    (out / 'results').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'runner.log')])
    guard = (out / 'run.lock').open('w')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)

    if args.items:
        pool, excluded = [json.loads(l) for l in args.items.read_text().splitlines()], None
    else:
        pool, excluded = load_pool(args.n, args.seed)
    snapdir = out / 'code_snapshot'
    snapdir.mkdir(exist_ok=True)
    srcs = ['run_minio3_replica.py', 'minio3_prompts.py', 'behavior_prompts.py', 'monitor_step.py', 'trajgen_models.py', 'probe_replica_api.py',
            'probe_api.py', 'run_api_t0.py']
    for f in srcs:
        shutil.copy2(ROOT / 'obs/src' / f, snapdir / f)
    cfg = dict(vars(args), out=str(out), started=datetime.now(timezone.utc).isoformat(),
               source='Mini-o3/VisualProbe_train', excluded_test_image_overlap_scanned=excluded,
               max_pixels=MAX_PIXELS, min_side=MIN_SIDE, max_upscale=MAX_UPSCALE,
               items_sha256=sha_file(args.items) if args.items else None,
               code_sha256={f: sha_file(snapdir / f) for f in srcs})
    (out / 'config.json').write_text(json.dumps(cfg, indent=1, default=str))
    (out / 'items.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in pool))

    effort = None if args.effort == 'none' else args.effort
    api = make_api(args.model, out, rpm=args.rpm, max_tokens=args.max_tokens, effort=effort)
    judge_api = make_api(args.judge_model, out, rpm=args.rpm, max_tokens=2048, effort=None)
    monitor = StepMonitor(make_api(args.monitor, out, rpm=args.rpm, max_tokens=3000, effort=None)) if args.monitor else None
    prompts = (system_prompt_v1, turn_instruction_v1) if args.prompt_set == 'behavior_v1' else None
    args.opts = dict(prompts=prompts, min_crop_px=args.min_crop_px, lenient=args.lenient_parse, monitor=monitor,
                     max_revisions=args.max_revisions, trajectory_revision_cap=args.trajectory_revision_cap)
    results = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_one, api, judge_api, it, out, args): it for it in pool}
        for f in as_completed(futs):
            it = futs[f]
            try:
                r = f.result()
                results.append(r)
                LOG.info('%s term=%s ground=%d judge=%s direct=%s', it['doc_id'], r['rollout']['termination'],
                         r['rollout']['n_grounding'], (r['judge_rollout'] or {}).get('score'),
                         (r['judge_direct'] or {}).get('score'))
            except Exception:  # keep other items running; the receipt stays in calls/
                LOG.exception('item failed: %s', it['doc_id'])
    kept = [to_coldstart(r, r['rollout']) for r in results
            if r['rollout']['termination'] == 'final' and (r['judge_rollout'] or {}).get('score') == 1]
    (out / 'sft_coldstart_format.jsonl').write_text(''.join(json.dumps(k, ensure_ascii=False) + '\n' for k in kept))
    status = dict(finished=datetime.now(timezone.utc).isoformat(), n_items=len(pool), n_done=len(results), n_kept=len(kept))
    (out / 'status.json').write_text(json.dumps(status, indent=1))
    LOG.info('done %s', status)


if __name__ == '__main__':
    main()
