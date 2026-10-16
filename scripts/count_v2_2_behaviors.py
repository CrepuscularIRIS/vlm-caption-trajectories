"""Count literature-listed behaviors in the V2.2 4,040-path SFT export.

Keyword and crop-coordinate counts only; nothing here looks at images.
Input is caption_sft_v2_2_keep.json as written by reconstruct_v2_2_export.py.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

REGION = re.compile(r"Its region in original_image \(normalized\): \[([^\]]+)\]")
SOURCE = re.compile(r'"source":"([^"]+)"')
WORD = re.compile(r"Step \d+ \| (\w+)")
LACKS = re.compile(
    r"\b(not (visible|shown|in (this|the) (crop|view))|does not (show|include|contain)|doesn't show"
    r"|no (visible )?\w+ (is )?visible|cut off|outside (the|this) (crop|view)|missed|misses"
    r"|only (partly|partially)|partial(ly)?|truncat|too (small|blurry|tight)|still unclear"
    r"|not (clear|legible|readable))",
    re.I,
)
CONTRAST = re.compile(r"\brather than\b", re.I)


def iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def flags(conv: list[dict]) -> dict[str, bool]:
    gpt = [m["value"] for m in conv if m["from"] == "gpt"]
    human = [m["value"] for m in conv if m["from"] == "human"]
    words = [w.group(1) for w in map(WORD.search, gpt) if w]
    regions = [list(map(float, m.group(1).split(","))) for m in map(REGION.search, human) if m]
    sources = [s.group(1) for s in map(SOURCE.search, gpt) if s]
    return {
        "direct_one_step": len(gpt) == 1,
        "narrow_crop_from_observation": any(s.startswith("observation") for s in sources),
        "re_examine": "re_examine" in words,
        "fallback": "fallback" in words,
        "abstain": "abstain" in words,
        "says_view_lacks_then_crops": any(
            LACKS.search(g) and "<grounding>" in g for g in gpt[1:]
        ),
        "two_disjoint_crops_iou_lt_0.1": any(
            iou(regions[i], regions[j]) < 0.1 for i in range(len(regions)) for j in range(i)
        ),
        "adjacent_crops_iou_lt_0.5": any(
            iou(regions[i - 1], regions[i]) < 0.5 for i in range(1, len(regions))
        ),
        "contrast_rather_than": any(CONTRAST.search(g) for g in gpt[1:]),
        "budget_used_up": any("This is the final turn" in h for h in human),
        "crops_ge_4": len(regions) >= 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path, help="caption_sft_v2_2_keep.json")
    rows = json.loads(parser.parse_args().export.read_text())
    counts: Counter = Counter()
    steps: Counter = Counter()
    for row in rows:
        conv = row["conversations"]
        counts.update(k for k, v in flags(conv).items() if v)
        steps[sum(m["from"] == "gpt" for m in conv)] += 1
    n = len(rows)
    report = {k: {"paths": v, "share": round(v / n, 4)} for k, v in sorted(counts.items())}
    print(json.dumps({"paths": n, "behaviors": report, "assistant_steps": dict(sorted(steps.items()))}, indent=2))


if __name__ == "__main__":
    main()
