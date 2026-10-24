"""CPU-only descriptive checks of the published DL summary; no model or new samples.

This does not reconstruct training gradients, identify causes, or test the new head.
Run from any cwd. The source hash is included so input drift remains visible.
"""
import hashlib
import json
import math
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / 'logs/dl/compare_dl.json'
raw = SOURCE.read_bytes()
data = json.loads(raw)
rows = data['per_state']
n = len(rows)

def mean(values):
    return sum(values) / n

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

zctrl = [math.log(r['p_ctrl']) - math.log1p(-r['p_ctrl']) for r in rows]
lo, hi = -100.0, 100.0
# Unique minimizer of cross entropy with p_main as the soft target.
for _ in range(100):
    b = (lo + hi) / 2
    if sum(sigmoid(z + b) - r['p_main'] for z, r in zip(zctrl, rows)) > 0:
        hi = b
    else:
        lo = b
bias = (lo + hi) / 2
oracle = mean(max(r['delta'], 0) for r in rows)
threshold_results = {}
for key in ('p_ref', 'p_ctrl'):
    thresholds = [-1.0] + sorted({r[key] for r in rows})
    scores = [mean(r['delta'] for r in rows if r[key] > t) for t in thresholds]
    best = max(scores)
    threshold_results[key] = {
        'best_same_table_threshold_score': best,
        'same_table_oracle_score': oracle,
        'same_table_oracle_minus_threshold': oracle - best,
        'epsilon_0_1_scaled_gap': .1 * (oracle - best),
    }

out = {
    'status': 'descriptive_reanalysis_not_causal_test',
    'source': 'logs/dl/compare_dl.json',
    'source_sha256': hashlib.sha256(raw).hexdigest(),
    'n': n,
    'U_choice': mean((r['p_main'] - r['p_ctrl']) * r['delta'] for r in rows),
    'p_increase_vs_ref': {
        str(e): sum(r['p_main'] - r['p_ref'] > e for r in rows)
        for e in (0, 1e-6, .001, .01)
    },
    'finite_main_logits': sum(0 < r['p_main'] < 1 for r in rows),
    'audit_BCE_derivative_of_common_logit_offset': {
        k: mean(abs(r['delta']) * r['p_' + k] - max(r['delta'], 0) for r in rows)
        for k in ('ref', 'main', 'ctrl')
    },
    'constant_logit_offset_description': {
        'fit_criterion': 'cross_entropy(p_main, sigmoid(logit(p_ctrl)+b)); no reward labels used',
        'b': bias,
        'probability_rmse': math.sqrt(mean((sigmoid(z+b) - r['p_main'])**2 for z, r in zip(zctrl, rows))),
        'unchanged_control_probability_rmse': math.sqrt(mean((r['p_ctrl'] - r['p_main'])**2 for r in rows)),
        'strict_rank_reversals': sum(
            (r['p_ctrl']-s['p_ctrl']) * (r['p_main']-s['p_main']) < 0
            for i, r in enumerate(rows) for s in rows[i+1:]
        ),
    },
    'threshold_and_oracle_descriptions': threshold_results,
    'limits': [
        'Audit-state scalar derivatives are not actual training gradients or optimizer steps.',
        'Constant-offset fitting is descriptive, not a causal decomposition.',
        'Threshold and oracle both use the same noisy K4 audit gaps: not a population upper bound or held-out test.',
        'No claim of accuracy non-inferiority, training mechanism identification, or new-head effectiveness.',
    ],
}
print(json.dumps(out, indent=2))
