"""Offline scorer for VisReason open / bbox answers the corpus scorer leaves unscored (2026-10-07).

open: comma-separated parts, lowercased, non-alphanumerics removed, compared as sets (strict; paraphrases count as wrong).
bbox: same number of boxes, greedy one-to-one match, every GT box needs IoU >= 0.5. Returns True / False / None (unparseable).
"""
import json
import re


def _norm(s) -> str:
    return re.sub(r'[^a-z0-9,]', '', str(s).lower())


def open_ok(ans, gold) -> bool | None:
    if ans is None:
        return None
    a = {x for x in _norm(ans).split(',') if x}
    g = {x for x in _norm(gold).split(',') if x}
    return bool(a) and a == g


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def bbox_ok(ans, gold) -> bool | None:
    try:
        a = json.loads(ans) if isinstance(ans, str) else ans
    except (ValueError, TypeError):
        return None
    if not isinstance(a, list) or not gold:
        return None
    if a and not isinstance(a[0], list):
        a = [a]
    g = gold if isinstance(gold[0], list) else [gold]
    if len(a) != len(g):
        return False
    used = set()
    for gb in g:
        best = max(((_iou(ab, gb), j) for j, ab in enumerate(a) if j not in used), default=(0.0, None))
        if best[0] < .5:
            return False
        used.add(best[1])
    return True


def rescore(result: dict) -> int | None:
    """1 / 0 / None using the stored score when present, else the offline open/bbox rule."""
    s = result['score'].get('score')
    if s is not None:
        return s
    if result['rollout']['termination'] != 'answer':
        return None
    spec = result['item']['answer_spec']
    ok = (open_ok(result['rollout']['answer'], spec.get('gold')) if spec['kind'] == 'open'
          else bbox_ok(result['rollout']['answer'], spec.get('gold')) if spec['kind'] == 'bbox' else None)
    return None if ok is None else int(ok)
