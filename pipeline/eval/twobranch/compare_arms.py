"""Matched-step comparison of the dV arm vs the original arm (mentor step 4).
E1/E2 per step (40/80/120/160): accuracy per benchmark and overall, crops/q, direct share (no crop), generated tokens;
paired item-bootstrap 95% CI of dv-orig accuracy and crops; E2 accuracy. Training rollouts: first-turn crop share by
dV bin (dV >= .5, |dV| < .5, dV <= -.5) per 40-step window for both arms (dV looked up from dv_table.json by uid is
not available in the rollout log for the orig arm, so bins come from the dv arm log's 'dv' field and, for orig, from
the per-step question uid mapping of the shared data order; both arms use the same data seed, so step s shows the same
questions)."""
import collections, json, os, random, sys  # [scrub] os added
d = sys.argv[1]
def load(t):
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
out = {"E1": {}, "E2": {}}
for st in (40, 80, 120, 160):
    for e, sfx in (("E1", ""), ("E2", "_perturb")):
        A, B = load(f"dv{st}{sfx}"), load(f"orig{st}{sfx}")
        if A and B:
            out[e][st] = dict(dv=summ(A), orig=summ(B), d_acc=paired(A, B, "ok"),
                              d_crops=paired([dict(r, ok=r["n_crops"]) for r in A], [dict(r, ok=r["n_crops"]) for r in B], "ok"))
def rows(path):
    return [json.loads(l) for l in open(path)]
RD = rows(os.environ["OUTPUT_DIR"] + "/grpo_v3_A_dv_s42_rollouts.jsonl")
RO = rows(os.environ["OUTPUT_DIR"] + "/grpo_v3_A_floor_orig_s42_rollouts.jsonl")
q2dv = {r["question"]: r["dv"] for r in RD if r.get("question") and r.get("dv") is not None}
def roll(R, dvf):
    # first action = crop iff the rollout made >= 1 crop (same proxy for both arms; the orig log has no 'first' field)
    w = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in R:
        dv = dvf(r)
        if dv is None: continue
        b = "pos" if dv >= .5 else "neg" if dv <= -.5 else "mid"
        k = (r["step"] - 1) // 40; c = w[k][b]; c[0] += 1; c[1] += int(r["crops"] > 0)
    return {f"s{40*k+1}-{40*k+40}": {b: dict(n=v[0], first_crop=v[1] / v[0]) for b, v in bins.items()} for k, bins in sorted(w.items())}
out["train_first_crop"] = dict(dv=roll(RD, lambda r: r.get("dv")), orig=roll(RO, lambda r: q2dv.get(r.get("question"))),
                               orig_unmapped=sum(r.get("question") not in q2dv for r in RO))
json.dump(out, open(f"{d}/compare.json", "w"), indent=1)
print(json.dumps(out, indent=1))
