"""Wave-1 comparison of the decision-state-baseline arms vs the RLOO controls at matched steps (80, 160).
Arms: db (DB), crop (crop-only). Controls: r1 (RLOO lambda 0.01, the card's recipe) and r0 (RLOO lambda 0).
E1/E2: accuracy per benchmark and overall, crops/q, direct share, generated tokens; paired item-bootstrap 95% CI
(2,000, seed 0) of arm - r1 and arm - r0 accuracy and crops, and db - crop. Training rollouts per 40-step window:
direct share, crops, accuracy; per-step [db_forks] / [db_credit] log means (forks, truncation, mean b, E[c^2] new vs RLOO).
Usage: python compare_db.py <db_eval_dir> <rloo_eval_dir> <outputs_dir> <out.json>"""
import collections, json, random, re, sys
dd, rd, outs, dst = sys.argv[1:5]
ARMS = {"db": "grpo_v3_A_db_db", "crop": "grpo_v3_A_db_crop"}
def load(d, t):
    try: return [json.loads(l) for l in open(f"{d}/traj_{t}.jsonl")]
    except FileNotFoundError: return None
def summ(R):
    n = len(R); by = collections.defaultdict(list)
    for r in R: by[r["source"]].append(r["ok"])
    return dict(n=n, acc=sum(r["ok"] for r in R) / n, crops=sum(r["n_crops"] for r in R) / n,
                followups=sum(max(0, r["n_crops"] - 1) for r in R) / n,
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
out = {"E1": {}, "E2": {}, "train": {}, "logs": {}}
for st in (80, 160):
    for e, sfx in (("E1", ""), ("E2", "_perturb")):
        R = {a: load(dd, f"{a}{st}{sfx}") for a in ARMS}
        r1, r0 = load(rd, f"r1_{st}{sfx}"), load(rd, f"r0_{st}{sfx}")
        if not (r1 and r0): continue
        cell = dict(r1=summ(r1), r0=summ(r0))
        for a, Ra in R.items():
            if Ra:
                cell[a] = summ(Ra); cell[f"{a}_vs_r1"] = pair(Ra, r1); cell[f"{a}_vs_r0"] = pair(Ra, r0)
        if R["db"] and R["crop"]: cell["db_vs_crop"] = pair(R["db"], R["crop"])
        out[e][st] = cell
for a, p in ARMS.items():
    try: Rr = [json.loads(l) for l in open(f"{outs}/{p}_rollouts.jsonl")]
    except FileNotFoundError: continue
    w = collections.defaultdict(list)
    for r in Rr: w[(r["step"] - 1) // 40].append(r)
    out["train"][a] = {f"s{40*k+1}-{40*k+40}": dict(n=len(v), direct=sum(r["crops"] == 0 for r in v) / len(v),
                       crops=sum(r["crops"] for r in v) / len(v), acc=sum(float(r["correct"]) for r in v) / len(v))
                       for k, v in sorted(w.items())}
    vals = collections.defaultdict(list)
    for line in open(f"{outs}/{p}.log", errors="ignore"):
        for key in ("forks", "truncated", "decision_states", "mean_b", "Ec2_new", "Ec2_rloo", "followup_turns", "tokens_changed"):
            m = re.search(rf"\b{key}=([0-9.]+)", line) if ("[db_forks]" in line or "[db_credit]" in line) else None
            if m: vals[key].append(float(m.group(1)))
    out["logs"][a] = {k: dict(n=len(v), mean=sum(v) / len(v)) for k, v in vals.items() if v}
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps(out)[:3000])
