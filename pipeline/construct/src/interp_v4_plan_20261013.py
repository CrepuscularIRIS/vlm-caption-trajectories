"""Construction v4 (skill-card protocol, forced interpolation on the generator's own requests): planning, no I/O.

What changes against v3r2 (2026-10-12):
  * five disturbance kinds instead of three, each tied to the skill it is meant to elicit:
      shift (adjacent, no overlap)      -> shift
      clip  (target cut by the frame)   -> shift
      tight (inside the target, context lost) -> widen
      wide  (target inside a much larger frame, too small again) -> narrow
      far   (a different part of the parent view) -> widen / locate again
    Pixel Reasoner inserts a zero-overlap box (40% of image data) or a too-large containing box (30%)
    before the correct crop; Mini-o3 keeps the generator's own near misses. Here the miss is derived from
    the generator's own request, so it is a plausible near miss, not a random box;
  * more disturbed decisions per task (0-4, slots 1-6) and a higher chance of two misses in a row;
  * a per-task crop budget of 8, 12 or 16 stated by the environment, so remaining budget is a real input;
  * per-task delivery scale 672 or 448 (longer side), so the same kind of question sometimes needs a crop
    and sometimes does not (Adaptive-CoF labels need-for-zoom by resolution).
Everything is drawn from a hash of the task id. Nothing reads GT, earlier answers or future views.
"""
import hashlib
import json
import re
from typing import List, Optional, Sequence, Tuple

from monitor_step import expand_min_crop

VERSION = 'interp-v4-skill-card-20261013'
ARM_VERIFY, ARM_PLAIN = 'verify_each_fact_v6', 'plain_v6'
DEFAULT_VARIANT = dict(
    name='v4_mix',
    arms={ARM_VERIFY: .70, ARM_PLAIN: .30},
    interventions={0: .12, 1: .36, 2: .30, 3: .15, 4: .07},
    slots={1: .30, 2: .24, 3: .18, 4: .13, 5: .09, 6: .06},
    view_max_side={672: .70, 448: .30},
    budget={8: .35, 12: .40, 16: .25},
    kinds=dict(shift=.28, clip=.24, tight=.14, wide=.18, far=.16),
    chain=0.25,
    conditions=None)
WIDE_SCALE = 2.6
WIDE_MIN_AREA_RATIO = 3.0
SHIFT_GAP = .10
CLIP_STEP = .60
TIGHT_SCALE = .45
FAR_TRIES = 12
# Share of the intended (effective, native-pixel) region still inside the executed one.
COVERAGE_LIMITS = dict(shift=(0.0, .15), clip=(.15, .65), tight=(0.0, .50), wide=(.95, 1.0), far=(0.0, 0.0))
ELICITS = dict(shift='shift', clip='shift', tight='widen', wide='narrow', far='widen')
MIN_NATIVE_PX = 56
MIN_BOX = .01
RESERVED_RECOVERY_CROPS = 2
GROUNDING = re.compile(r'<grounding>.*?</grounding>', re.S)
Box = Sequence[float]

VERIFY = """Trajectory construction condition (teacher side; the student never sees this text).
Evidence is confirmed by inspection, not by impression. Views are delivered at reduced size, so treat fine detail in [original_image] as unconfirmed until a crop shows it at a usable scale. Before you commit, every fact the answer depends on must have been inspected in a received view: each reading, each object identity or attribute, each row, column or legend binding, each counted region. Take one open fact or region per crop. Do not settle several distant details with one broad crop, and do not crop what the question does not depend on. For counting and comparison, sweep the relevant regions with scan and keep the tally in KNOW. When a reading decides the answer and is still tentative, use verify once before committing.
The crop tool can return a region other than the one you asked for. The SEE line must report what the returned pixels show for the detail you wanted, with the honest verdict, including when the view missed or cut it. Then choose the skill that fits what went wrong: shift when you can tell which way the target lies, widen when the context is lost, narrow when the target is there but too small. Never describe content that is not in the returned pixels.
There is no target length. A question that depends on one fact needs one crop; one that depends on many needs many. Do not invent uncertainty, errors or observations, and do not pad.

Style reference for single turns. These show wording only; the content is unrelated to your task and must not be reused.

Step 1 | locate
SEE: [original_image] shows a street with a bus stop sign at the left kerb; its lettering is not legible at this size.
KNOW: nothing yet.
NEXT: the route number on the sign; the sign is the anchor, so I take the region around it.

Step 2 | shift
SEE: miss - [observation_1] shows the pole and pavement below the sign, not the sign face.
KNOW: nothing yet.
NEXT: the sign face lies above this view, so I move the region up on [original_image].

Step 3 | narrow
SEE: partial - [observation_2] contains the sign face but the digits are a few pixels tall.
KNOW: the sign has one line of digits [observation_2].
NEXT: those digits; I crop the digit line inside [observation_2].

Step 4 | scan
SEE: hit - [observation_3] shows two cups on the left shelf, both white.
KNOW: left shelf 2 white cups [observation_3]; middle and right shelves not covered.
NEXT: the middle shelf, the next uncovered region.

Step 5 | verify
SEE: hit - [observation_4] reads 4?7 with the middle digit touching the frame edge.
KNOW: first digit 4, last digit 7 [observation_4]; middle digit tentative.
NEXT: the middle digit, which decides between two options; I take a slightly larger region around it.

Step 6 | commit
SEE: hit - [observation_5] shows the middle digit as 1.
KNOW: the number is 417 [observation_5].
SO: 417 matches option B."""
CONDITIONS = {ARM_VERIFY: VERIFY, ARM_PLAIN: ''}


def unit(*parts: object) -> float:
    """Deterministic uniform number in [0, 1) from the construction version and ``parts``."""
    text = json.dumps([VERSION, *parts], sort_keys=True, default=str)
    return int(hashlib.sha256(text.encode()).hexdigest()[:13], 16) / 16 ** 13


def weighted(options: Sequence[Tuple[object, float]], u: float) -> object:
    total = sum(w for _, w in options)
    acc = 0.0
    for value, weight in options:
        acc += weight / total
        if u < acc:
            return value
    return options[-1][0]


def _options(weights: dict) -> List[Tuple[object, float]]:
    return sorted(((int(k) if str(k).isdigit() else k, w) for k, w in weights.items()), key=lambda kv: str(kv[0]))


def _draw(value: object, audit_id: str, name: str) -> object:
    return weighted(_options(value), unit(audit_id, name)) if isinstance(value, dict) else value


def make_plan(audit_id: str, variant: Optional[dict] = None) -> dict:
    """Per-task arm, budget, delivery scale and the ordinals of the generator's own crop requests to disturb."""
    variant = variant or DEFAULT_VARIANT
    count = weighted(_options(variant['interventions']), unit(audit_id, 'count'))
    pool = _options(variant['slots'])
    slots: List[int] = []
    for i in range(min(count, len(pool))):
        slot = weighted(pool, unit(audit_id, 'slot', i))
        slots.append(slot)
        pool = [(s, w) for s, w in pool if s != slot]
    return dict(version=VERSION, variant=variant['name'], arm=_draw(variant['arms'], audit_id, 'arm'),
                planned_interventions=len(slots), slots=sorted(slots),
                view_max_side=_draw(variant.get('view_max_side'), audit_id, 'side'),
                budget=_draw(variant['budget'], audit_id, 'budget'),
                kinds=variant['kinds'], chain=variant.get('chain') or 0.0)


def condition(variant: dict, arm: str) -> str:
    return (variant.get('conditions') or {}).get(arm, CONDITIONS[arm])


def due(slots: Sequence[int], ordinal: int, used: int, forced_run: int, crops_before: int, budget: int,
        chain_draw: float = 1.0, chain: float = 0.0) -> bool:
    """Disturb this crop request? Only while a recovery crop still fits the budget; never three in a row."""
    if budget - crops_before < RESERVED_RECOVERY_CROPS:
        return False
    if forced_run:
        return forced_run == 1 and chain_draw < chain
    return sum(s <= ordinal for s in slots) - used > 0


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _effective(box: Box, parent_px: Box, width: int, height: int) -> Tuple[float, ...]:
    """Native-pixel region the environment would really cut for a normalized box of a view."""
    pw, ph = parent_px[2] - parent_px[0], parent_px[3] - parent_px[1]
    req = (parent_px[0] + box[0] * pw, parent_px[1] + box[1] * ph, parent_px[0] + box[2] * pw, parent_px[1] + box[3] * ph)
    eff, _ = expand_min_crop(req, width, height, MIN_NATIVE_PX)
    return tuple(float(round(x)) for x in eff)


def _candidates(box: Box, kind: str, seed: Sequence[object]) -> List[Tuple[str, List[float]]]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if kind == 'tight':
        return [('center', [cx - w * TIGHT_SCALE / 2, cy - h * TIGHT_SCALE / 2, cx + w * TIGHT_SCALE / 2, cy + h * TIGHT_SCALE / 2])]
    if kind == 'wide':
        nw, nh = min(1.0, w * WIDE_SCALE), min(1.0, h * WIDE_SCALE)
        if nw * nh < WIDE_MIN_AREA_RATIO * w * h:
            return []
        ax, ay = min(max(0.0, cx - nw / 2), 1 - nw), min(max(0.0, cy - nh / 2), 1 - nh)
        return [('around', [ax, ay, ax + nw, ay + nh])]
    if kind == 'far':
        if w > .6 or h > .6:
            return []
        out = []
        for i in range(FAR_TRIES):
            ax, ay = unit(*seed, 'far_x', i) * (1 - w), unit(*seed, 'far_y', i) * (1 - h)
            if abs(ax + w / 2 - cx) >= 1.5 * w or abs(ay + h / 2 - cy) >= 1.5 * h:
                out.append((f'elsewhere_{i}', [ax, ay, ax + w, ay + h]))
        return out
    step = 1 + SHIFT_GAP if kind == 'shift' else CLIP_STEP
    moves = (('left', -step * w, 0), ('right', step * w, 0), ('up', 0, -step * h), ('down', 0, step * h))
    return [(name, [x0 + dx, y0 + dy, x1 + dx, y1 + dy]) for name, dx, dy in moves]


def perturb(box: Box, parent_px: Box, width: int, height: int, seed: Sequence[object], kinds: dict) -> Optional[dict]:
    """Replace the generator's own requested region by a missed, clipped, over-tight, over-wide or distant one.

    Geometry is tested on the regions the environment would really cut. It bounds how much of the
    intended region remains visible; it does not certify that the needed evidence is absent.
    """
    intended = _effective(box, parent_px, width, height)
    if not _area(intended):
        return None
    kinds = {k: w for k, w in kinds.items() if w > 0}
    order = sorted(kinds, key=lambda k: unit(*seed, 'kind', k) / kinds[k])
    for kind in order:
        low, high = COVERAGE_LIMITS[kind]
        valid = []
        for direction, cand in _candidates(box, kind, seed):
            cand = [round(v, 4) for v in cand]
            if not (0 <= cand[0] and cand[2] <= 1 and 0 <= cand[1] and cand[3] <= 1):
                continue
            if cand[2] - cand[0] < MIN_BOX or cand[3] - cand[1] < MIN_BOX:
                continue
            eff = _effective(cand, parent_px, width, height)
            inter = _area([max(eff[0], intended[0]), max(eff[1], intended[1]), min(eff[2], intended[2]), min(eff[3], intended[3])])
            coverage = inter / _area(intended)
            if kind == 'wide' and _area(eff) < WIDE_MIN_AREA_RATIO * _area(intended):
                continue
            if low <= coverage <= high and eff != intended:
                valid.append(dict(kind=kind, direction=direction, bbox=cand, intended_region_coverage=round(coverage, 4),
                                  elicits=ELICITS[kind]))
        if valid:
            return min(valid, key=lambda c: unit(*seed, 'direction', c['direction']))
    return None


def forced_text(turn_text: str, bbox: Box, source: str) -> str:
    """The generator's own turn with only the executed region replaced."""
    action = '<grounding>' + json.dumps(dict(bbox_2d=list(bbox), source=source), separators=(',', ':')) + '</grounding>'
    text, n = GROUNDING.subn(lambda _: action, turn_text, count=1)
    if n != 1:
        raise ValueError('Expected exactly one grounding action')
    return text
