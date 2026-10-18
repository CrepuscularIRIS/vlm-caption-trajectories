"""Per-trajectory behaviour features for eval_v4 records (pure Python, no GPU).

A record is one line of traj_<tag>[_perturb].jsonl. Boxes are absolute pixels; boxes_abs[0] is the full image.
"""
from typing import Optional

from traj_utils import iou

ZOOM_CONTAIN = 0.9  # a crop is "inside" another when >= 90% of its area lies in it
RELOC_IOU = 0.5
RETURN_IOU = 0.3


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inside(a, b) -> bool:
    """True when >= ZOOM_CONTAIN of box a lies inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return _area(a) > 0 and ix / _area(a) >= ZOOM_CONTAIN


def transition(prev, nxt) -> str:
    """Classify consecutive crops: zoom_in (next inside previous), widen (previous inside next), shift (IoU < 0.5
    and neither contains the other), or stay."""
    if _inside(nxt, prev) and _area(nxt) < _area(prev):
        return "zoom_in"
    if _inside(prev, nxt) and _area(nxt) > _area(prev):
        return "widen"
    return "shift" if iou(prev, nxt) < RELOC_IOU else "stay"


def features(r: dict) -> dict:
    crops = r["boxes_abs"][1:]
    moves = [transition(a, b) for a, b in zip(crops, crops[1:])]
    sources = [t["parsed"].get("source") for t in r["turns"] if t["parsed"].get("kind") == "grounding"
               and "box_abs" in t]
    verdicts = [t["parsed"].get("verdict") for t in r["turns"]]
    return dict(
        ok=r["ok"], ok_train=r.get("ok_train_rule", r["ok"]), n_crops=r["n_crops"],
        direct=int(r["n_crops"] == 0), multi=int(r["n_crops"] >= 2),
        reloc=int(any(m in ("shift", "widen") for m in moves)), shift=int("shift" in moves),
        widen=int("widen" in moves), zoom_in=int("zoom_in" in moves),
        back_to_original=int("original_image" in sources[1:]),
        invalid=int(any(t.get("check_errors") for t in r["turns"])),
        overlong=int(bool(r["overlong"])), error=int(bool(r["error"])),
        budget_out=int(r["n_crops"] >= 8), gen_tokens=r["gen_tokens"],
        said_miss=int(any(v in ("miss", "partial") for v in verdicts if v)),
    )


def recovery(pert: dict, base: Optional[dict]) -> Optional[dict]:
    """E2 features for one perturbed record, paired with the unperturbed record of the same (uid, sample)."""
    p = pert.get("perturb") or {}
    if not p.get("shown_bbox") or not p.get("ok"):
        return None
    w, h = pert["image_wh"]
    a = p["asked_bbox"]
    intended = [a[0] * w, a[1] * h, a[2] * w, a[3] * h]
    later = pert["boxes_abs"][2:]  # crops after the perturbed one
    recrop = int(len(later) > 0)
    return dict(
        ok=pert["ok"], ok_base=None if base is None else base["ok"], recrop=recrop,
        back_to_target=int(any(iou(b, intended) >= RETURN_IOU for b in later)),
        false_confirm=int(not recrop and not pert["ok"]),
        extra_crops=None if base is None else pert["n_crops"] - base["n_crops"],
        said_miss=int(any(t["parsed"].get("verdict") in ("miss", "partial") for t in pert["turns"][1:2])),
    )
