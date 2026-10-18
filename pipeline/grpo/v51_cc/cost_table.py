"""Training-cost columns per GRPO run from its console log (via log_metrics.py CSV rows).

updates      = optimizer steps (global_batch_size == rollout_batch_size, one update per step)
gen tokens   = sum over steps of timing_s/gen * 1000 / timing_per_token_ms/gen  (policy-sampled, loss-masked tokens)
proc tokens  = sum perf/total_num_tokens (all tokens through the actor, incl. prompts, receipts, images)
wall / GPU-h = sum perf/time_per_step; GPU-h = wall * n_gpus
Usage: python cost_table.py RUN.log [RUN.log ...]   (run from outputs/)
"""
import csv, io, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent

def rows(log):
    out = subprocess.run([sys.executable, str(HERE / "log_metrics.py"), log], capture_output=True, text=True, check=True).stdout
    return [r for r in csv.DictReader(io.StringIO(out)) if int(r["step"]) > 0]

def f(r, k):
    v = r.get(k)
    return float(v) if v not in (None, "") else 0.0

print("| run | updates | gen tokens (M) | processed tokens (M) | step time (h) | GPU-h (2 GPU) |")
print("|---|---|---|---|---|---|")
for log in sys.argv[1:]:
    R = rows(log)
    keys = R[0].keys()
    tk = next(k for k in keys if k.endswith("time_per_step"))
    nk = next(k for k in keys if k.endswith("total_num_tokens"))
    gk = next(k for k in keys if k.endswith("timing_s/gen"))
    pk = next(k for k in keys if k.endswith("timing_per_token_ms/gen"))
    gen = sum(f(r, gk) * 1000 / f(r, pk) for r in R if f(r, pk) > 0)
    wall = sum(f(r, tk) for r in R)
    print(f"| {log[:-4]} | {len(R)} | {gen/1e6:.2f} | {sum(f(r, nk) for r in R)/1e6:.1f} | {wall/3600:.2f} | {2*wall/3600:.1f} |")
