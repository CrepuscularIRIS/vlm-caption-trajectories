"""Plot saved metrics only; continuation is separate from equal-budget baselines."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'logs/stopped40_20261029'
base = json.loads((ROOT / 'logs/k8_20261028/curves_k8.json').read_text())
extra = json.loads((OUT / 'curves_continuation.json').read_text())
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
panels = [('reward/accuracy', 'Training accuracy (logged)'),
          ('val/accuracy_reward', 'Development accuracy (422 questions)'),
          ('response_length/mean', 'Response length incl. tool receipts'),
          ('actor/grad_norm', 'Gradient norm before clipping'),
          ('timing_s/step', 'Seconds / step'),
          ('timing_s/gen', 'Seconds / generation')]
colors = dict(grpo='C0', dapo='C1', gspo='C2', gigpo='C3', base_grpo='C4')
for ax, (metric, title) in zip(axes.flat, panels):
    for run, metrics in base.items():
        name = run.removeprefix('k8_')
        points = sorted((int(s), d[metric]) for s,d in metrics.items() if metric in d)
        if points:
            ax.plot(*zip(*points), color=colors[name], label=name+' (0-32)', lw=1.2,
                    marker='o' if metric.startswith('val/') else None, ms=3)
    for name, metrics in extra.items():
        points = sorted((int(s), d[metric]) for s,d in metrics.items() if metric in d)
        if points:
            ax.plot(*zip(*points), color=colors[name], linestyle='--', lw=1.5,
                    label=name+' (extra budget)', marker='x', ms=3)
    ax.axvline(32, color='grey', lw=.8)
    ax.set(title=title, xlabel='Cumulative training step');ax.grid(alpha=.25)
axes[0,0].legend(fontsize=8)
fig.suptitle('4K x 8 baselines and stopped continuation; evaluation checkpoint = 40 for dashed runs')
fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=150)
plt.close(fig)
