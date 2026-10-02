"""Caption v5.1 runner on corpus_v1 items (real crop environment, one generator per run directory).

Differences from run_caption_mvp.py (kept unchanged for the frozen v3/v4 runs):
- items come from obs/benchmarks/corpus_v1 (item_id, question, image, answer_spec, material, split);
- explicit crop budget 0..8 (--budget), passed to the first turn, receipts and hard checks;
- every view records source pixels, the size actually sent, its parent, the executed box and sha256;
- generation shows per-item hints; the export is the student view (same SYSTEM, no hints) and the
  export manifest records that removal;
- answers are scored with corpus_scoring (deterministic; None = not scorable, never "wrong");
- --effort sends reasoning_effort (K3 low vs default is checked on the shared smoke items).
Raw API receipts are written by RoleAPI under OUT/calls; nothing is rewritten after the fact.
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import caption_v51 as proto
from corpus_scoring import score
from monitor_step import expand_min_crop
from probe_api import text_msg
from run_minio3_replica import ROOT, resolve_source, save_obs, sha_file

LOG = logging.getLogger('caption_v51')
MIN_CROP_PX = 56


def view_record(view_id: str, parent: str | None, box: tuple, src_img: Image.Image, path: Path, W: int, H: int) -> dict:
    shown = Image.open(path).size
    return dict(view_id=view_id, parent=parent, effective_box_px=list(box),
                fov_original=[box[0] / W, box[1] / H, box[2] / W, box[3] / H],
                source_px=list(src_img.size), shown_px=list(shown), sha256=sha_file(path), path=str(path))


def call_turn(gen, item, n, msgs, max_tokens, temperature, top_p, available, crops, budget, paths, format_retries):
    rejected = []
    for attempt in range(format_retries + 1):
        rid = item['item_id'] + '#v51' + (f'#fmt{attempt}' if attempt else '')
        r = gen.call(f"{item['item_id']}/s{n}" + (f'/fmt{attempt}' if attempt else ''), msgs, max_tokens,
                     replicate_id=rid, temperature=temperature, top_p=top_p)
        raw = r.get('raw_text') or r.get('content') or ''
        if r.get('finish_reason') == 'tool_calls' and not raw.strip() and attempt < format_retries:
            rejected.append(dict(raw_text=raw, violations=['native_tool_call_empty'], request_sha256=r.get('request_sha256')))
            continue
        if r.get('finish_reason') != 'stop':
            return r, raw, None, [], rejected
        p = proto.parse_step(raw)
        if hasattr(gen, 'coord_mode') and p.get('kind') == 'grounding':
            k_src = available.index(p['source']) if p.get('source') in available else None
            size = gen.model_input_size(str(paths[k_src])) if k_src is not None else None
            p = proto.to_unit_bbox(p, gen.coord_mode, size)
        viol = proto.check_step(p, n, crops, available, budget)
        if not viol or attempt == format_retries:
            return r, raw, p, viol, rejected
        rejected.append(dict(raw_text=raw, violations=viol, request_sha256=r.get('request_sha256')))
    return r, raw, None, [], rejected


def rollout(gen, item: dict, odir: Path, budget: int, temperature: float, top_p: float, max_tokens: int,
            format_retries: int = 2) -> dict:
    hint_keys = proto.task_hints(item)
    full = Image.open(item['image']).convert('RGB')
    W, H = full.size
    boxes = [(0.0, 0.0, float(W), float(H))]
    paths = [save_obs(full, odir / 'obs0.jpg')]
    views = [view_record('original_image', None, boxes[0], full, paths[0], W, H)]
    available = ['original_image']
    head, tail = proto.first_user_text(item['question'], hint_keys, views[0]['source_px'], views[0]['shown_px'], budget)
    api_tags = not getattr(gen, 'supports_prefill', False)
    view = proto.to_api_tags if api_tags else (lambda t: t)
    msgs = [text_msg('system', view(proto.SYSTEM)),
            {'role': 'user', 'content': [{'type': 'text', 'text': head}, gen.image(str(paths[0])),
                                         {'type': 'text', 'text': tail}]}]
    turns, steps, term, crops = [], [], None, 0
    for n in range(1, budget + 2):
        r, raw, p, viol, rejected = call_turn(gen, item, n, msgs, max_tokens, temperature, top_p, available, crops,
                                              budget, paths, format_retries)
        step = dict(step=n, request_sha256=r.get('request_sha256'), finish_reason=r.get('finish_reason'),
                    usage=r.get('usage'), raw_text=raw, crops_before=crops, available_before=list(available),
                    rejected_format_samples=rejected)
        steps.append(step)
        if r.get('finish_reason') == 'error':
            term = 'api_error'
            break
        if p is None:
            term = 'truncated'
            break
        step.update(parsed={k: v for k, v in p.items() if k != 'errors'}, violations=viol, flags=proto.soft_flags(p))
        turns.append({'role': 'gpt', 'text': p.get('text', raw)})
        if viol:
            term = 'protocol_violation'
            break
        if p['kind'] == 'answer':
            term = 'abstain' if p['answer'].upper() == 'UNCLEAR' else 'answer'
            break
        k = resolve_source(p['source'], len(boxes))
        x0, y0, x1, y1 = boxes[k]
        b = p['bbox']
        req = (x0 + b[0] * (x1 - x0), y0 + b[1] * (y1 - y0), x0 + b[2] * (x1 - x0), y0 + b[3] * (y1 - y0))
        ab, expanded = expand_min_crop(req, W, H, MIN_CROP_PX)
        ab = tuple(float(round(v)) for v in ab)
        crop = full.crop(tuple(int(v) for v in ab))          # always re-cut from the full-resolution original
        crops += 1
        obs_id = f'observation_{len(boxes)}'
        paths.append(save_obs(crop, odir / f'obs{len(boxes)}.jpg'))
        if ab in boxes[1:]:
            step['flags'].append('repeat_crop')
        boxes.append(ab)
        vr = view_record(obs_id, available[k], ab, crop, paths[-1], W, H)
        views.append(vr)
        available.append(obs_id)
        rh, rt = proto.receipt_text(n, len(boxes) - 1, p['source'], vr['fov_original'], vr['source_px'],
                                    vr['shown_px'], available, crops, budget)
        step.update(source_index=k, requested_box=[round(v, 1) for v in req], expanded=expanded, observation=obs_id)
        turns.append({'role': 'human', 'head': rh, 'tail': rt})
        msgs += [{'role': 'assistant', 'content': view(p.get('text', raw))},
                 {'role': 'user', 'content': [{'type': 'text', 'text': rh}, gen.image(str(paths[-1])),
                                              {'type': 'text', 'text': rt}]}]
    answer = steps[-1].get('parsed', {}).get('answer') if term in ('answer', 'abstain') else None
    rel = [str(q.relative_to(ROOT)) if str(q).startswith(str(ROOT)) else str(q) for q in paths]
    export = None
    if term in ('answer', 'abstain'):
        export = proto.to_sharegpt(item['question'], turns, rel, views[0]['source_px'], views[0]['shown_px'], budget)
    return dict(protocol=proto.PROTOCOL_VERSION, hint_keys=hint_keys, budget=budget, termination=term, answer=answer,
                crops=crops, steps=steps, views=views, export=export,
                export_manifest=dict(teacher_scaffold_removed=['hints'] if hint_keys else [],
                                     system_identical=True, budget_shown=budget))


def run_item(gen, item: dict, out: Path, a) -> dict:
    dest = out / 'results' / f"{item['item_id']}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    odir = out / 'obs' / item['item_id']
    odir.mkdir(parents=True, exist_ok=True)
    ro = rollout(gen, item, odir, a.budget, a.temperature, a.top_p, a.max_tokens, a.format_retries)
    sc = score(item['answer_spec'], ro['answer']) if ro['termination'] == 'answer' else {'score': 0 if ro['termination'] == 'abstain' else None, 'method': ro['termination']}
    res = dict(item=item, model=a.model, effort=a.effort, rollout=ro, score=sc)
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    tmp.replace(dest)
    return res


def main() -> None:
    from trajgen_models import make_api
    p = argparse.ArgumentParser()
    p.add_argument('--items', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--budget', type=int, default=proto.MAX_CROPS)
    p.add_argument('--effort', default=None, choices=[None, 'low', 'medium', 'high'])
    p.add_argument('--format-retries', type=int, default=2)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--ids', nargs='*', default=None, help='restrict to these item_ids')
    a = p.parse_args()
    a.out, a.items = a.out.resolve(), a.items.resolve()
    (a.out / 'results').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(a.out / 'runner.log')])
    (a.out / 'config.json').write_text(json.dumps(dict(vars(a), protocol=proto.PROTOCOL_VERSION, min_crop_px=MIN_CROP_PX,
                                                       started=datetime.now(timezone.utc).isoformat()), indent=1, default=str))
    gen = make_api(a.model, a.out, max_tokens=a.max_tokens, effort=a.effort)
    items = [json.loads(line) for line in a.items.read_text().splitlines()]
    if a.ids:
        items = [it for it in items if it['item_id'] in set(a.ids)]
    for it in items:
        r = run_item(gen, it, a.out, a)
        ro = r['rollout']
        LOG.info('%s term=%s crops=%d answer=%s score=%s behaviors=%s', it['item_id'], ro['termination'], ro['crops'],
                 ro['answer'], r['score']['score'], [s.get('parsed', {}).get('behavior') for s in ro['steps']])


if __name__ == '__main__':
    main()
