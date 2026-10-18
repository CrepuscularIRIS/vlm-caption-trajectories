"""Per-benchmark E1 table from eval_v4 per-sample outputs (traj_<tag>.jsonl): accuracy (ok), crops, gen tokens."""
import collections, glob, json, sys

def acc(R, key="ok"):
    return sum(r[key] for r in R) / len(R)

files = sys.argv[1:] or sorted(f for f in glob.glob("traj_*.jsonl") if "perturb" not in f)
srcs = None
for f in files:
    rows = [json.loads(l) for l in open(f)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    srcs = srcs or sorted(by)
    n = len(rows)
    print(f"| {f[5:-6]} | " + " | ".join(f"{acc(by[s]):.3f}" for s in srcs) + f" | {acc(rows):.3f} | {acc(rows,'ok_train_rule'):.3f} | "
          f"{sum(r['n_crops'] for r in rows)/n:.2f} | {sum(r['n_crops']==0 for r in rows)/n:.2f} | {sum(r['gen_tokens'] for r in rows)/n:.0f} |")
print("columns:", srcs, {s: len(by[s]) for s in srcs})
