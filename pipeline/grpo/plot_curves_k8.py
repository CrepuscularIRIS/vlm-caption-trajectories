"""Training curves of the five 4k x 8 runs from logs/k8_20261028/curves_k8.json.

curves_k8.json = per-step EasyR1 log metrics (obs train_curves parser) plus two fields merged from private run files:
`crops/mean` (mean crops over all generated rollouts of the step; DAPO includes filtered groups, its generation calls
mapped to steps with dapo_chunks_per_step.json) and `adv/nonzero` (non-zero advantage token share, updates/*.jsonl).
usage: python plot_curves_k8.py logs/k8_20261028/curves_k8.json out.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"k8_grpo": "SFT + GRPO", "k8_dapo": "SFT + DAPO", "k8_gspo": "SFT + GSPO", "k8_gigpo": "SFT + GiGPO",
         "k8_base_grpo": "base + GRPO"}
PANELS = [("reward/accuracy", "train accuracy (logged)"), ("val/accuracy_reward", "val accuracy (422 q)"),
          ("crops/mean", "crops / trajectory (all generated rollouts)"), ("response_length/mean", "response tokens (mean)"),
          ("adv/nonzero", "non-zero advantage token share (update)"), ("timing_s/step", "seconds / step")]

d = json.load(open(sys.argv[1]))
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for ax, (key, title) in zip(axes.flat, PANELS):
    for run, label in NAMES.items():
        pts = sorted((int(s), m[key]) for s, m in d[run].items() if key in m)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o" if key.startswith("val") else ".",
                ms=4, lw=1.2, label=label)
    ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=.3)
axes.flat[0].legend(fontsize=8)
fig.suptitle("4k pool (unique images) x n=8, 32 steps; A-SFT / base start; single seed")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130)
