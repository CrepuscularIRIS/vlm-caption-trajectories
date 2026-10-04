"""Funnel v1 selection, CPU stage: per-trajectory features, geometry operations, per-item consensus,
difficulty and routing (L1 pass / L2 / L3). No model calls.

v2 (2026-10-03, after the Codex quality review): geometry describes the crop OPERATION only
(v5.1: "Geometry alone does not determine a behavior"). A widen/switch/new-region crop issued with a
'previous view was inadequate' cue is only a fallback CANDIDATE for review, never a validated fallback.
Self-label/geometry mismatch, repeat crops and commits that do not cite a crop are audit flags, not hard
rejections. Hard gates: correct answer, no unresolved coordinates, no coordinate talk in captions.

(v1 notes, kept for history)

Behavior labels are derived from crop geometry and citations, not from the generator's own step word:
  locate        first crop, or a crop of a new region taken from the original image
  zoom_in       new view lies inside the previous crop
  adjust        new view overlaps the previous crop substantially (shift / small resize)
  widen         new view contains the previous crop and is clearly larger
  switch        new view barely overlaps the previous crop
  fallback      widen/switch right after a caption that reported missing or insufficient evidence
A self-declared word that contradicts the geometry (e.g. 'fallback' on a zoom_in) is counted as a label mismatch.
Difficulty comes from the models' own decisions: who answered directly, who needed crops, who was right.

Incremental: features are cached per result file (keyed by mtime), so a patrol run only parses new files.
"""
import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

SELECT_VERSION = 'v2'
FAMILY = {'gpt-6-luna': 'openai', 'gpt-6.1-sol': 'openai', 'gpt-6-sol': 'openai', 'gpt-6-astra': 'openai',
          'glm-5.3-flash': 'zhipu', 'glm-5.3': 'zhipu', 'grok-4.7': 'xai', 'k3-256k': 'moonshot',
          'kimi-for-coding': 'moonshot',
          'gemini-3.8-flash-low': 'google', 'claude-sonnet-5-5': 'anthropic'}
# Cue that the PREVIOUS VIEW failed (crop inadequate), not that the searched object is absent:
# "no pedestrians visible" in an absence scan is a finding, not a reason to fall back.
MISSING_CUE = re.compile(r"\b(miss(ed|es)|cut off|clipped|(outside|beyond) (the|this) (crop|view|frame)|"
                         r"not (visible|shown|included) in (this|the) (crop|view)|not in (this|the) crop|"
                         r"too (tight|narrow|small|blurr?y|close|zoomed)|blurr?(ed|y)|pixelated|cannot (see|read|tell)|"
                         r"unreadable|illegible|lacks? (the )?context|insufficient)\b", re.I)
OVER_ZOOM = 4.0  # shown/source linear scale above which detail is mostly interpolation


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def geometry_op(box, prev, source_is_original):
    """Crop operation relative to the previous crop (both in original-image fractions). Description only."""
    if prev is None:
        return 'locate'
    a, p, i = area(box), area(prev), inter(box, prev)
    if a <= 0 or p <= 0:
        return 'adjust'
    if i / a >= 0.9 and a < 0.7 * p:
        return 'zoom_in'
    if i / p >= 0.9 and a >= 1.5 * p:
        return 'widen'
    if i / min(a, p) < 0.2:
        return 'locate' if source_is_original else 'switch'
    return 'adjust'


def fallback_candidate(op: str, is_first: bool, caption_text: str) -> bool:
    """Audit trigger: a non-zoom move issued right after the model reported the last view as inadequate."""
    return not is_first and op in ('widen', 'switch', 'locate') and bool(MISSING_CUE.search(caption_text or ''))


SELF_CONFLICT = {'fallback': {'zoom_in'}}


def trajectory_features(r: dict) -> dict:
    ro, item = r['rollout'], r['item']
    steps, views = ro['steps'], {v['view_id']: v for v in ro['views']}
    ops, fb_cands, mismatches, flags = [], 0, 0, Counter()
    prev_box, prev_caption, cited = None, '', set()
    holds = revises = 0
    tokens = prompt_max = completion = 0
    for s in steps:
        u = s.get('usage') or {}
        tokens += u.get('total_tokens') or 0
        prompt_max = max(prompt_max, u.get('prompt_tokens') or 0)
        completion += u.get('completion_tokens') or 0
        p = s.get('parsed') or {}
        cited.update(p.get('cited') or [])
        for up in p.get('updates') or []:
            holds += up.get('kind') == 'HOLD'
            revises += up.get('kind') == 'REVISE'
        for f in s.get('flags') or []:
            flags[f] += 1
        if p.get('kind') == 'grounding' and s.get('observation') in views:
            v = views[s['observation']]
            op = geometry_op(v['fov_original'], prev_box, p.get('source') == 'original_image')
            ops.append(op)
            # The caption that issues a crop reports what the latest view missed; the cue can sit there or one step earlier.
            fb_cands += fallback_candidate(op, prev_box is None, (p.get('caption') or '') + ' ' + prev_caption)
            if op in SELF_CONFLICT.get(p.get('behavior'), set()):
                mismatches += 1
            sw, sh = v['source_px']
            dw, dh = v['shown_px']
            if sw and dw / sw > OVER_ZOOM:
                flags['over_zoom'] += 1
            prev_box = v['fov_original']
        prev_caption = p.get('caption') or ''
    crops = ro['crops']
    last = (steps[-1].get('parsed') or {}) if steps else {}
    commit_cites = set(last.get('cited') or [])
    obs_ids = [v for v in views if v != 'original_image']
    uncited = [v for v in obs_ids if v not in cited]
    return dict(item_id=item['item_id'], bench=item['bench'], material=item['material'], group_id=item['group_id'],
                model=r['model'], family=FAMILY.get(r['model'], r['model']), effort=r.get('effort'),
                demo=r.get('demo_kind'), term=ro['termination'], answer=ro['answer'],
                score=r['score'].get('score'), crops=crops, n_steps=len(steps), geometry_ops=ops,
                fallback_candidates=fb_cands,
                self_words=[(s.get('parsed') or {}).get('behavior') for s in steps],
                label_mismatch=mismatches, holds=holds, revises=revises, flags=dict(flags),
                commit_cites_crop=bool(commit_cites & set(obs_ids)), uncited_crops=len(uncited),
                format_retries=sum(len(s.get('rejected_format_samples') or []) for s in steps),
                normalized=len(ro.get('format_normalized_requests') or []),
                truncated=ro['termination'] == 'truncated' or any(s.get('finish_reason') == 'length' for s in steps),
                tokens=tokens, prompt_tokens_max=prompt_max, completion_tokens=completion, wall_s=r.get('wall_s'))


def gate(t: dict) -> list[str]:
    """Hard CPU gates for a clean candidate. Empty list = passes."""
    why = []
    if t['term'] != 'answer' or t['score'] != 1:
        why.append('not_correct_answer')
    if t['flags'].get('coordinate_unresolved'):
        why.append('coordinate_unresolved')
    if t['flags'].get('coordinate_talk'):  # v5.1: coordinates stay in the action; captions describe what is seen
        why.append('coordinate_talk')
    return why


def audit_flags(t: dict) -> list[str]:
    """Reasons to send a candidate to semantic review first. Not rejections."""
    why = []
    if t['crops'] and not t['commit_cites_crop']:
        why.append('commit_does_not_cite_crop')
    if t['label_mismatch']:
        why.append('self_word_vs_geometry')
    if t['flags'].get('repeat_crop'):
        why.append('repeat_crop')
    if t['flags'].get('over_zoom'):
        why.append('over_zoom')
    if t.get('fallback_candidates'):
        why.append('fallback_candidate')
    if t['revises']:
        why.append('revise_present')
    if t['normalized']:
        why.append('format_normalized')
    return why


def item_summary(trajs: list[dict]) -> dict:
    answered = [t for t in trajs if t['term'] in ('answer', 'abstain')]
    right = [t for t in trajs if t['score'] == 1]
    fam_right = {t['family'] for t in right}
    direct_right = [t for t in right if t['crops'] == 0]
    wrong_answers = Counter(t['answer'] for t in answered if t['term'] == 'answer' and t['score'] == 0)
    wrong_fams = defaultdict(set)
    for t in answered:
        if t['term'] == 'answer' and t['score'] == 0:
            wrong_fams[t['answer']].add(t['family'])
    gold_suspect = any(len(f) >= 2 for f in wrong_fams.values()) and len(fam_right) <= 1
    n = len({t['family'] for t in trajs})
    l2_tried = any(t.get('stage') == 'l2' for t in trajs)
    if n < 2:
        difficulty, route = 'pending', 'pending'
    elif len({t['family'] for t in direct_right}) >= 2:
        difficulty, route = 'easy', 'L1'
    elif len(fam_right) >= 2:
        difficulty, route = 'medium', 'L1'
    elif len(fam_right) == 1:
        difficulty, route = 'hard', 'L2'
    else:
        difficulty, route = 'unsolved', 'L2'
    # L3 (user + Astra/Opus) only after the L2 strong model also failed to give a cross-family consensus.
    # A correct L2 trajectory goes to Sonnet review (L2_review); an L2 attempt that is still wrong goes to L3.
    if route == 'L2' and l2_tried:
        route = 'L2_review' if any(t.get('stage') == 'l2' and t['score'] == 1 for t in trajs) else 'L3'
    direct_wrong = any(t['crops'] == 0 and t['term'] == 'answer' and t['score'] == 0 for t in trajs)
    crop_right = any(t['crops'] > 0 for t in right)
    return dict(item_id=trajs[0]['item_id'], bench=trajs[0]['bench'], material=trajs[0]['material'],
                group_id=trajs[0]['group_id'], models=sorted(t['model'] for t in trajs), families=n,
                correct_models=sorted(t['model'] for t in right), direct_correct_models=sorted(t['model'] for t in direct_right),
                difficulty=difficulty, route=route, gold_suspect=gold_suspect,
                observation_needed=direct_wrong and crop_right, top_wrong=wrong_answers.most_common(1))


def rank_key(t: dict, item: dict):
    """Within an item: direct first for easy items, crop trajectories first when observation was needed;
    then fewer soft flags, fewer steps, fewer tokens (token efficiency)."""
    prefer_crop = item['observation_needed'] or item['difficulty'] in ('medium', 'hard')
    soft = (sum(t['flags'].get(k, 0) for k in ('no_citation', 'over_zoom', 'long_caption')) + t['uncited_crops']
            + len(audit_flags(t)))
    return ((t['crops'] == 0) == prefer_crop, soft, t['n_steps'], t['tokens'])


def load(root: Path, cache_path: Path) -> list[dict]:
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            cache[row['path']] = row
    out, new = [], []
    for p in sorted(root.glob('l[12]_*/results/*.json')):
        m = p.stat().st_mtime
        row = cache.get(str(p))
        if not row or row['mtime'] != m:
            f = trajectory_features(json.loads(p.read_text()))
            f['stage'] = p.parent.parent.name[:2]
            row = dict(path=str(p), mtime=m, f=f)
            new.append(row)
        out.append(row)
    if new:
        with cache_path.open('a') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in new)
    return [r['f'] for r in out]


def build(root: Path) -> dict:
    sel = root / f'select_{SELECT_VERSION}'
    sel.mkdir(exist_ok=True)
    trajs = load(root, sel / 'traj_cache.jsonl')
    by_item = defaultdict(list)
    for t in trajs:
        by_item[t['item_id']].append(t)
    items, cands = [], []
    for iid, ts in by_item.items():
        it = item_summary(ts)
        items.append(it)
        ok = sorted((t for t in ts if not gate(t)), key=lambda t: rank_key(t, it)) if it['route'] == 'L1' else []
        for rank, t in enumerate(ok):
            cands.append(dict(t, rank=rank, difficulty=it['difficulty'], observation_needed=it['observation_needed'],
                              audit_flags=audit_flags(t), select_version=SELECT_VERSION))
    write_lines(sel / 'items.jsonl', items)
    write_lines(sel / 'l1_candidates.jsonl', cands)
    write_lines(sel / 'queue_L2.jsonl', [i for i in items if i['route'] == 'L2'])
    write_lines(sel / 'queue_L3.jsonl', [i for i in items if i['route'] == 'L3'])
    summary = summarize(trajs, items, cands)
    (sel / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def write_lines(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    tmp.replace(path)


def summarize(trajs, items, cands) -> dict:
    per_model = {}
    for m in sorted({t['model'] + '_' + str(t['effort']) for t in trajs}):
        ts = [t for t in trajs if t['model'] + '_' + str(t['effort']) == m]
        ans = [t for t in ts if t['term'] == 'answer']
        crop_steps = Counter(l for t in ts for l in t['geometry_ops'])
        gates = Counter(g for t in ts for g in gate(t))
        per_model[m] = dict(
            n=len(ts), terminations=dict(Counter(t['term'] for t in ts)),
            accuracy=round(sum(t['score'] == 1 for t in ans) / len(ans), 3) if ans else None,
            direct_rate=round(sum(t['crops'] == 0 for t in ans) / len(ans), 3) if ans else None,
            mean_crops=round(st.mean(t['crops'] for t in ans), 2) if ans else None,
            geometry_ops=dict(crop_steps), fallback_candidates=sum(t['fallback_candidates'] for t in ts),
            label_mismatch=sum(t['label_mismatch'] for t in ts),
            holds=sum(t['holds'] for t in ts), revises=sum(t['revises'] for t in ts),
            gate_failures=dict(gates), truncated=sum(t['truncated'] for t in ts),
            format_retries=sum(t['format_retries'] for t in ts), normalized=sum(t['normalized'] > 0 for t in ts),
            median_tokens=st.median(t['tokens'] for t in ts), max_prompt_tokens=max(t['prompt_tokens_max'] for t in ts),
            tokens_per_correct=round(sum(t['tokens'] for t in ts) / max(1, sum(t['score'] == 1 for t in ts))))
    return dict(trajectories=len(trajs), items=len(items), per_model=per_model,
                difficulty=dict(Counter(i['difficulty'] for i in items)), route=dict(Counter(i['route'] for i in items)),
                observation_needed=sum(i['observation_needed'] for i in items),
                gold_suspect=sum(i['gold_suspect'] for i in items),
                l1_candidates=len(cands), l1_items_with_candidate=len({c['item_id'] for c in cands}),
                candidate_geometry_ops=dict(Counter(l for c in cands if c['rank'] == 0 for l in c['geometry_ops'])),
                candidate_audit_flags=dict(Counter(f for c in cands if c['rank'] == 0 for f in c['audit_flags'])),
                top_candidate_direct=sum(c['crops'] == 0 for c in cands if c['rank'] == 0),
                by_bench_difficulty={b: dict(Counter(i['difficulty'] for i in items if i['bench'] == b))
                                     for b in sorted({i['bench'] for i in items})})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.root.resolve()), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
