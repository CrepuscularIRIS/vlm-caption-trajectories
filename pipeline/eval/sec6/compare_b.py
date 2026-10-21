"""Direction B matched-step comparison: arms b1 (replace_first_centred) and b2 (add_all_centred) vs the 10-20 orig arm
(same seed, data order and recipe). Same statistics as compare_arms.py: E1/E2 per step (40/80/120/160) accuracy per
benchmark, crops/q, direct share, generated tokens; paired item-bootstrap 95% CI (2,000, seed 0) of arm - orig accuracy
and crops; training first-turn crop share (proxy: crops > 0, as for orig) by dV bin per 40-step window.
Usage: python compare_b.py <b_eval_dir> <orig_eval_dir> <outputs_dir> <out.json>"""
import collections, json, random, sys
bd, od, outs, dst = sys.argv[1:5]
ARMS = {"b1": "grpo_v3_A_dv_b1_s42", "b2": "grpo_v3_A_dv_b2_s42"}
def load(d, t):
    try: return [json.loads(l) for l in open(f"{d}/traj_{t}.jsonl")]
    except FileNotFoundError: return None
def summ(R):
    n = len(R); by = collections.defaultdict(list)
    for r in R: by[r["source"]].append(r["ok"])
    return dict(n=n, acc=sum(r["ok"] for r in R) / n, crops=sum(r["n_crops"] for r in R) / n,
                direct=sum(r["n_crops"] == 0 for r in R) / n, tok=sum(r["gen_tokens"] for r in R) / n,
                by_source={s: sum(v) / len(v) for s, v in sorted(by.items())})
def paired(A, B, key):
    qa, qb = collections.defaultdict(list), collections.defaultdict(list)
    for r in A: qa[r["uid"]].append(r[key])
    for r in B: qb[r["uid"]].append(r[key])
    U = sorted(set(qa) & set(qb)); diffs = [sum(qa[u]) / len(qa[u]) - sum(qb[u]) / len(qb[u]) for u in U]
    rng = random.Random(0); bs = sorted(sum(rng.choice(diffs) for _ in U) / len(U) for _ in range(2000))
    return dict(mean=sum(diffs) / len(U), lo=bs[50], hi=bs[1949])
out = {"E1": {}, "E2": {}, "train_first_crop": {}}
for arm in ARMS:
    for st in (40, 80, 120, 160):
        for e, sfx in (("E1", ""), ("E2", "_perturb")):
            A, B = load(bd, f"{arm}{st}{sfx}"), load(od, f"orig{st}{sfx}")
            if A and B:
                out[e].setdefault(arm, {})[st] = dict(arm=summ(A), orig=summ(B), d_acc=paired(A, B, "ok"),
                    d_crops=paired([dict(r, ok=r["n_crops"]) for r in A], [dict(r, ok=r["n_crops"]) for r in B], "ok"))
def roll(path):
    w = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    try: R = [json.loads(l) for l in open(path)]
    except FileNotFoundError: return None
    for r in R:
        dv = r.get("dv")
        if dv is None: continue
        b = "pos" if dv >= .5 else "neg" if dv <= -.5 else "mid"
        k = (r["step"] - 1) // 40; c = w[k][b]; c[0] += 1; c[1] += int(r["crops"] > 0)
    return {f"s{40*k+1}-{40*k+40}": {b: dict(n=v[0], first_crop=v[1] / v[0]) for b, v in bins.items()} for k, bins in sorted(w.items())}
for arm, p in ARMS.items():
    out["train_first_crop"][arm] = roll(f"{outs}/{p}_rollouts.jsonl")
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps(out)[:3000])
