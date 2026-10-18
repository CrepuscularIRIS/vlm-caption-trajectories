"""Construction v3 (intent-miss interpolation): pure planning and geometry, no I/O.

Why a new version. In the 2026-10-11 constructor the editor first wrote an
already-visible key fact into the forced caption and then cropped away from it,
always at step 2. The fact stayed in the history, so answering right away was
reasonable and 89.5% of interpolated candidates became the same three-step path.

Here the intervention is placed where the generator itself has just declared
something unresolved: it asked for a crop. The caption of that turn is kept
verbatim (the generator's own statement of what is missing); only the executed
region is displaced, clipped or over-tightened. Nothing that the missing view
would have shown is written anywhere, so the gap is real until a later crop
closes it. Which grounding decisions are disturbed (none, early, middle, late,
several) is drawn per task from a fixed hash, never from GT, old answers or
future views.

r2 (after pilot 1): variants are data (a JSON file frozen with the campaign), a fourth
disturbance ``wide`` returns the asked region inside a much larger one so the detail is
too small again, and a disturbed crop may with a set probability be followed by a second
disturbed one (a two-miss chain).
"""
import hashlib
import json
import re
from typing import List, Optional, Sequence, Tuple

from monitor_step import expand_min_crop

VERSION = 'interp-v3r2-intent-miss-20261012'
ARM_VERIFY, ARM_PLAIN, ARM_COARSE = 'verify_each_fact', 'plain_v51', 'coarse_to_fine'
DEFAULT_VARIANT = dict(
    name='verify_miss',
    arms={ARM_VERIFY: .90, ARM_PLAIN: .10},
    interventions={0: .10, 1: .50, 2: .28, 3: .12},
    slots={1: .40, 2: .30, 3: .20, 4: .10},
    # None keeps the 2 MP delivery of the earlier databases; a number caps the longer side of
    # every delivered view (original and crops), so detail has to be reached by zooming.
    view_max_side=None,
    kinds=dict(shift=.45, clip=.35, tight=.20),
    # Probability that the crop right after a disturbed one is disturbed too (at most once in a row).
    chain=0.0,
    conditions=None)
WIDE_SCALE = 2.6
WIDE_MIN_AREA_RATIO = 3.0
SHIFT_GAP = .10
CLIP_STEP = .60
TIGHT_SCALE = .45
# Share of the intended (effective, native-pixel) region still inside the executed one.
COVERAGE_LIMITS = dict(shift=(0.0, .15), clip=(.15, .65), tight=(0.0, .50), wide=(.95, 1.0))
MIN_NATIVE_PX = 56
MIN_BOX = .01
RESERVED_RECOVERY_CROPS = 2
GROUNDING = re.compile(r'<grounding>.*?</grounding>', re.S)
Box = Sequence[float]

JUDGE = """The crop tool can be imprecise: a returned view may miss, clip or over-tighten the region you meant. Judge every returned view by its actual pixels and its reported region. Start each caption with what the latest view really shows for the detail you asked for, including when it missed or cut it; then name the single detail that is still unresolved and choose a different or corrected region for it. Never describe content that is not in the returned pixels, and keep facts that earlier views already support."""
STOP = """Stop as soon as every needed fact has been inspected. A question that depends on one fact needs one crop; one that depends on many needs many. There is no target length. Do not invent uncertainty, errors or observations."""
VERIFY_HEAD = """Trajectory construction condition: evidence is confirmed by inspection, not by impression.
Treat fine detail in [original_image] as unconfirmed until a crop shows it at a usable scale. Before committing, every fact the answer depends on (each reading, each object identity or attribute, each row, column or legend binding, each counted region) must have been inspected in a received view. Inspect one unresolved fact or region per crop. Do not settle several distant details with one broad crop, and do not crop what the question does not depend on."""
COARSE_HEAD = """Trajectory construction condition: search from coarse to fine.
Views are delivered at limited resolution, so a small detail becomes legible only when its region fills the view. First take a region that should contain the needed evidence together with enough context to recognise it; then narrow step by step onto the deciding detail until it is legible, one unresolved detail per crop. When a narrower view loses the target or its context, return to a wider earlier view and choose again. Before committing, every fact the answer depends on (each reading, each object identity or attribute, each row, column or legend binding, each counted region) must be legible in a received view. Do not crop what the question does not depend on."""
CONDITIONS = {ARM_VERIFY: '\n'.join((VERIFY_HEAD, JUDGE, STOP)), ARM_COARSE: '\n'.join((COARSE_HEAD, JUDGE, STOP)), ARM_PLAIN: ''}


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


def make_plan(audit_id: str, variant: Optional[dict] = None) -> dict:
    """Per-task arm and the ordinals of the generator's own grounding decisions to disturb."""
    variant = variant or DEFAULT_VARIANT
    count = weighted(_options(variant['interventions']), unit(audit_id, 'count'))
    pool = _options(variant['slots'])
    slots: List[int] = []
    for i in range(min(count, len(pool))):
        slot = weighted(pool, unit(audit_id, 'slot', i))
        slots.append(slot)
        pool = [(s, w) for s, w in pool if s != slot]
    return dict(version=VERSION, variant=variant['name'], arm=weighted(_options(variant['arms']), unit(audit_id, 'arm')),
                planned_interventions=len(slots), slots=sorted(slots), view_max_side=variant.get('view_max_side'),
                kinds=variant.get('kinds') or DEFAULT_VARIANT['kinds'], chain=variant.get('chain') or 0.0)


def condition(variant: dict, arm: str) -> str:
    """Teacher-only construction text of an arm; a variant may carry its own wording."""
    return (variant.get('conditions') or {}).get(arm, CONDITIONS[arm])


def due(slots: Sequence[int], ordinal: int, used: int, forced_run: int, crops_before: int, budget: int,
        chain_draw: float = 1.0, chain: float = 0.0) -> bool:
    """Disturb this grounding decision? Only while a recovery crop still fits the budget.

    Right after one disturbed crop the recovery is executed as asked, except that with
    probability ``chain`` it is disturbed once more; never three in a row.
    """
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


def _candidates(box: Box, kind: str) -> List[Tuple[str, List[float]]]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if kind == 'tight':
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return [('center', [cx - w * TIGHT_SCALE / 2, cy - h * TIGHT_SCALE / 2, cx + w * TIGHT_SCALE / 2, cy + h * TIGHT_SCALE / 2])]
    if kind == 'wide':
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        nw, nh = min(1.0, w * WIDE_SCALE), min(1.0, h * WIDE_SCALE)
        if nw * nh < WIDE_MIN_AREA_RATIO * w * h:
            return []
        ax, ay = min(max(0.0, cx - nw / 2), 1 - nw), min(max(0.0, cy - nh / 2), 1 - nh)
        return [('around', [ax, ay, ax + nw, ay + nh])]
    step = 1 + SHIFT_GAP if kind == 'shift' else CLIP_STEP
    moves = (('left', -step * w, 0), ('right', step * w, 0), ('up', 0, -step * h), ('down', 0, step * h))
    return [(name, [x0 + dx, y0 + dy, x1 + dx, y1 + dy]) for name, dx, dy in moves]


def perturb(box: Box, parent_px: Box, width: int, height: int, seed: Sequence[object],
            kinds: Optional[dict] = None) -> Optional[dict]:
    """Displace, clip or over-tighten the generator's own requested region.

    Geometry is tested on the regions the environment would really cut. It bounds how much of
    the intended region remains visible; it does not certify that needed evidence is absent.
    """
    intended = _effective(box, parent_px, width, height)
    if not _area(intended):
        return None
    kinds = {k: w for k, w in (kinds or DEFAULT_VARIANT['kinds']).items() if w > 0}
    order = sorted(kinds, key=lambda k: unit(*seed, 'kind', k) / kinds[k])
    for kind in order:
        low, high = COVERAGE_LIMITS[kind]
        valid = []
        for direction, cand in _candidates(box, kind):
            cand = [round(v, 4) for v in cand]
            if not (0 <= cand[0] and cand[2] <= 1 and 0 <= cand[1] and cand[3] <= 1):
                continue
            if cand[2] - cand[0] < MIN_BOX or cand[3] - cand[1] < MIN_BOX:
                continue
            eff = _effective(cand, parent_px, width, height)
            inter = _area([max(eff[0], intended[0]), max(eff[1], intended[1]), min(eff[2], intended[2]), min(eff[3], intended[3])])
            coverage = inter / _area(intended)
            if low <= coverage <= high and eff != intended:
                valid.append(dict(kind=kind, direction=direction, bbox=cand, intended_region_coverage=round(coverage, 4)))
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
