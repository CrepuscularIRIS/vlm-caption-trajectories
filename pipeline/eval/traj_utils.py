"""Pure helpers for the trajectory exporter (no torch / vLLM, so they can be unit-tested anywhere)."""
import re
import zlib
from typing import Callable, Optional, Sequence

Box = Sequence[float]

# eight shift directions in (dx, dy) units of the box's own width / height
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
PERTURB_IOU_MAX = 0.1
_LAST_BBOX = re.compile(r'("bbox_2d"\s*:\s*)\[[^\]]*\](?=[^<]*</grounding>\s*$)')


def stable_seed(base: int, uid: str, sample_idx: int, round_idx: int) -> int:
    """Deterministic 31-bit seed per request, independent of batch composition and Python hash randomisation."""
    return (zlib.crc32(f"{uid}|{sample_idx}|{round_idx}".encode()) ^ base) & 0x7FFFFFFF


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def shift_box(box: Box, dx: float, dy: float) -> list[float]:
    """Shift a normalised box by (dx*width, dy*height), then slide it back inside [0, 1] keeping its size."""
    w, h = box[2] - box[0], box[3] - box[1]
    x0, y0 = box[0] + dx * w, box[1] + dy * h
    x0 = min(max(0.0, x0), max(0.0, 1.0 - w))
    y0 = min(max(0.0, y0), max(0.0, 1.0 - h))
    return [x0, y0, x0 + w, y0 + h]


def choose_perturbation(box: Box, to_abs: Callable[[Box], Box], first_dir: int) -> dict:
    """Pick the shifted box whose *absolute* crop (after the env's minimum-size expansion) overlaps the intended
    crop least. `first_dir` (0-7, from the seed) decides which direction is tried first, so A and B see the same
    perturbation for the same (uid, sample) whenever their first box is the same."""
    intended = to_abs(box)
    best: Optional[dict] = None
    for k in range(len(DIRECTIONS)):
        dx, dy = DIRECTIONS[(first_dir + k) % len(DIRECTIONS)]
        cand = shift_box(box, dx, dy)
        ov = iou(intended, to_abs(cand))
        if best is None or ov < best["iou_abs"]:
            best = {"bbox": cand, "iou_abs": ov, "dir": (dx, dy)}
        if ov < PERTURB_IOU_MAX:
            break
    best["ok"] = best["iou_abs"] < PERTURB_IOU_MAX
    return best


def rewrite_bbox(text: str, bbox: Box) -> Optional[str]:
    """Replace the box of the final <grounding> action, leaving every other character of the turn unchanged.
    Used by the perturbation so the history reads as if the model itself had asked for the shifted box, the same
    way the v4c library's induced misses are written. Returns None when the turn has no grounding action."""
    new = "[" + ",".join(str(round(float(v), 4)) for v in bbox) + "]"
    out, n = _LAST_BBOX.subn(lambda m: m.group(1) + new, text, count=1)
    return out if n == 1 else None
