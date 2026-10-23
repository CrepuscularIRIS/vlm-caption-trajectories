"""Descriptive reading of crops after the first one (docs/31). Reads eval trajectories (one sample per question per
condition), classifies every later crop by box geometry and the stated intent in its Step text, and counts crop
revisits, cap hits and what follows a confirm-type crop. Descriptive only: no counterfactual, so nothing here says a
crop was unnecessary.

Usage: python3 later_crops.py <traj_clean.jsonl> <traj_perturb.jsonl> <out.json>
"""
import ast
import collections
import json
import re
import sys

CAP = 7
INTENT = {
    "closer/enlarge/read": r"closer|enlarge|tighter|zoom|magnif|legib|read",
    "confirm/verify": r"confirm|verif|double-check|recheck",
    "other/rest/another": r"rest of|other|remaining|another",
    "wider/whole": r"wider|whole|entire|full view|overall",
    "compare/reference": r"compar|contrast|reference",
}


def parsed(t: dict) -> dict:
    p = t["parsed"]
    return p if isinstance(p, dict) else ast.literal_eval(p)


def box(t: dict):
    b = t.get("box_abs")
    return ast.literal_eval(b) if isinstance(b, str) else b


def area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b) -> float:
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b) -> float:
    i = inter(a, b)
    return i / max(1, area(a) + area(b) - i)


def geometry(p: dict, b, prev: list, full: float) -> str:
    """Relation of a later crop to the earlier crops (boxes in original-image pixels)."""
    if p.get("source", "").startswith("observation"):
        return "zoom_into_crop"
    if b and area(b) >= 0.85 * full:
        return "full_image"
    best = max(prev, key=lambda q: iou(b, q))
    if iou(b, best) > 0.8:
        return "repeat_same_region"
    if inter(b, best) >= 0.9 * area(best) and area(b) > area(best):
        return "widen"
    if inter(b, best) >= 0.9 * area(b):
        return "shrink_from_original"
    if iou(b, best) > 0.3:
        return "shifted_overlap"
    return "new_region"


def analyse(rows: list) -> dict:
    n = len(rows)
    geo, geo_ok, intent = collections.Counter(), collections.defaultdict(list), collections.Counter()
    confirm_then = collections.Counter()
    by_n = collections.defaultdict(list)
    revisit = collections.defaultdict(list)
    turn2 = collections.Counter()
    first_full = 0
    for r in rows:
        turns = r["turns"]
        W, H = r["image_wh"]
        g = [(i, t) for i, t in enumerate(turns) if parsed(t).get("kind") == "grounding" and box(t)]
        by_n[min(r["n_crops"], CAP)].append(r["ok"])
        if len(turns) > 1 and r["n_crops"] >= 1:
            turn2[str(parsed(turns[1]).get("behavior"))] += 1
        if g and area(box(g[0][1])) >= 0.85 * W * H:
            first_full += 1
        orig = [box(t) for _, t in g if parsed(t).get("source") == "original_image"]
        rv = any(iou(orig[j], orig[k]) > 0.8 for j in range(len(orig)) for k in range(j + 2, len(orig)))
        revisit[(min(r["n_crops"], CAP), rv)].append(r["ok"])
        for j, (i, t) in enumerate(g):
            if j == 0:
                continue
            p, txt = parsed(t), t["text"].lower()
            c = geometry(p, box(t), [box(x) for _, x in g[:j]], W * H)
            geo[c] += 1
            geo_ok[c].append(r["ok"])
            hits = [k for k, pat in INTENT.items() if re.search(pat, txt)]
            for k in hits:
                intent[k] += 1
            if "confirm/verify" in hits and i + 1 < len(turns):
                confirm_then[str(parsed(turns[i + 1]).get("kind"))] += 1
    later = sum(geo.values())
    return {
        "trajectories": n, "accuracy": sum(r["ok"] for r in rows) / n,
        "crops_per_q": sum(r["n_crops"] for r in rows) / n,
        "share_ge2_crops": sum(r["n_crops"] >= 2 for r in rows) / n,
        "cap_hits": sum(r["n_crops"] >= CAP for r in rows),
        "first_crop_full_image": first_full,
        "later_crops": later,
        "later_crops_in_cap_hit_traj": sum(max(0, min(r["n_crops"], 99) - 1) for r in rows if r["n_crops"] >= CAP),
        "acc_by_n_crops": {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in sorted(by_n.items())},
        "turn2_behavior": dict(turn2.most_common()),
        "later_geometry": {k: {"n": v, "share": v / later, "traj_acc": sum(geo_ok[k]) / v} for k, v in geo.most_common()},
        "later_intent_keywords": dict(intent.most_common()),
        "after_confirm_crop_next_kind": dict(confirm_then.most_common()),
        "revisit_nonadjacent_iou_gt_.8": {f"n={k[0]},{'revisit' if k[1] else 'none'}": {"n": len(v), "acc": sum(v) / len(v)}
                                          for k, v in sorted(revisit.items()) if k[0] >= 3},
    }


def main() -> None:
    clean, pert, dest = sys.argv[1:4]
    load = lambda p: [json.loads(line) for line in open(p) if line.strip()]
    out = {"clean": analyse(load(clean)), "e2_first_crop_shifted": analyse(load(pert))}
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
