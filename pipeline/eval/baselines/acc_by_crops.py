"""Accuracy by executed crop count, per E1 condition (reads trajectories/e1_rollouts/*.jsonl.gz).

Usage: python pipeline/eval/baselines/acc_by_crops.py [out.json]
Writes per-condition crops histogram and accuracy at each crop level (crops>=5 bucketed).
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "trajectories" / "e1_rollouts"
CONDS = ["B0", "B0-ans", "B1", "B2", "B3", "B3-ans", "B3-cot", "B3-zoom",
         "B4", "B5", "B6", "H1", "H2", "H2-6k",
         "K-GRPO", "K-DAPO", "K-GSPO", "K-GiGPO", "K-base", "X1", "X2"]

out = {}
for c in CONDS:
    p = D / f"{c}.jsonl.gz"
    if not p.exists():
        continue
    hist = defaultdict(int)
    acc = defaultdict(lambda: [0, 0])  # level -> [correct, n]
    with gzip.open(p) as f:
        for line in f:
            r = json.loads(line)
            lvl = min(r.get("crops", 0), 5)
            hist[lvl] += 1
            acc[lvl][0] += r["correct"]
            acc[lvl][1] += 1
    out[c] = {
        "n": sum(hist.values()),
        "crops_hist": {str(k): hist[k] for k in sorted(hist)},
        "acc_by_crops": {str(k): {"acc": round(v[0] / v[1], 4), "n": v[1]}
                         for k, v in sorted(acc.items())},
    }

dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "acc_by_crops.json"
dest.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"written {dest}")
for c, d in out.items():
    row = " ".join(f"{k}裁:{v['acc']:.3f}({v['n']})" for k, v in d["acc_by_crops"].items())
    print(f"{c:8s} {row}")
