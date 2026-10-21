"""Mentor 10-21 control: RLOO lambda 0 (r0) and 0.01 (r1) vs the 10-20 orig arm (GRPO, std division, lambda 0, same
seed/data/recipe) at steps 40/80/120/160, and r1 vs r0. E1/E2: accuracy per benchmark and overall, crops/q, direct share,
generated tokens; paired item-bootstrap 95% CI (2,000, seed 0) of accuracy and crops differences.
Training rollouts per 40-step window: direct share (crops == 0), mean crops, accuracy, for r0, r1, orig and (if its log
exists) the 10-17 GRPO crop-cost lambda 0.01 run that collapsed (stopped at step 56).
Usage: python compare_rloo.py <rloo_eval_dir> <orig_eval_dir> <outputs_dir> <out.json>"""
import collections, json, random, sys
rd, od, outs, dst = sys.argv[1:5]
def load(d, t):
    try: return [json.loads(l) for l in open(f"{d}/traj_{t}.jsonl")]
    except FileNotFoundError: return None
def summ(R):
    n = len(R); by = collections.defaultdict(list)
    for r in R: by[r["source"]].append(r["ok"])
    return dict(n=n, acc=sum(r["ok"] for r in R) / n, crops=sum(r["n_crops"] for r in R) / n,
                direct=sum(r["n_crops"] == 0 for r in R) / n, tok=sum(r["gen_tokens"] for r in R) / n,
                by_source={s: sum(v) / len(v) for s, v in sorted(by.items())})
def paired(A, B, key="ok"):
    qa, qb = collections.defaultdict(list), collections.defaultdict(list)
    for r in A: qa[r["uid"]].append(r[key])
    for r in B: qb[r["uid"]].append(r[key])
    U = sorted(set(qa) & set(qb)); diffs = [sum(qa[u]) / len(qa[u]) - sum(qb[u]) / len(qb[u]) for u in U]
    rng = random.Random(0); bs = sorted(sum(rng.choice(diffs) for _ in U) / len(U) for _ in range(2000))
    return dict(mean=sum(diffs) / len(U), lo=bs[50], hi=bs[1949])
def pair(A, B):
    return dict(d_acc=paired(A, B), d_crops=paired([dict(r, ok=r["n_crops"]) for r in A], [dict(r, ok=r["n_crops"]) for r in B]))
out = {"E1": {}, "E2": {}, "train": {}}
for st in (40, 80, 120, 160):
    for e, sfx in (("E1", ""), ("E2", "_perturb")):
        r0, r1, og = load(rd, f"r0_{st}{sfx}"), load(rd, f"r1_{st}{sfx}"), load(od, f"orig{st}{sfx}")
        if not (r0 and r1 and og): continue
        out[e][st] = dict(r0=summ(r0), r1=summ(r1), orig=summ(og), r0_vs_orig=pair(r0, og), r1_vs_orig=pair(r1, og), r1_vs_r0=pair(r1, r0))
def windows(path):
    try: R = [json.loads(l) for l in open(path)]
    except FileNotFoundError: return None
    w = collections.defaultdict(list)
    for r in R: w[(r["step"] - 1) // 40].append(r)
    return {f"s{40*k+1}-{40*k+40}": dict(n=len(v), direct=sum(r["crops"] == 0 for r in v) / len(v),
            crops=sum(r["crops"] for r in v) / len(v), acc=sum(float(r["correct"]) for r in v) / len(v)) for k, v in sorted(w.items())}
for name, p in (("r0", "grpo_v3_A_rloo_l000"), ("r1", "grpo_v3_A_rloo_l010"), ("orig", "grpo_v3_A_floor_orig_s42"), ("grpo_cost_l010", "grpo_v3_A_cropcost")):
    out["train"][name] = windows(f"{outs}/{p}_rollouts.jsonl")
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps(out)[:3000])
