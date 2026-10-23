"""Zero-credit control vs four-case arm (prep_fcz/PREREG.md). Primary: E1 step 160, fc - zero, paired item bootstrap
(2,000, seed 0) of accuracy and crops/q. Descriptive: zero vs r1, E2, step 80, training windows, zero's log means.
Usage: python compare_fcz.py <fc_eval_dir> <zero_eval_dir> <rloo_eval_dir> <outputs_dir> <out.json>"""
import collections
import json
import random
import re
import sys

fd, zd, rd, outs, dst = sys.argv[1:6]


def load(d, t):
    try:
        return [json.loads(line) for line in open(f"{d}/traj_{t}.jsonl")]
    except FileNotFoundError:
        return None


def summ(R):
    n = len(R)
    by = collections.defaultdict(list)
    for r in R:
        by[r["source"]].append(r["ok"])
    return dict(n=n, acc=sum(r["ok"] for r in R) / n, crops=sum(r["n_crops"] for r in R) / n,
                direct=sum(r["n_crops"] == 0 for r in R) / n, tok=sum(r["gen_tokens"] for r in R) / n,
                by_source={s: sum(v) / len(v) for s, v in sorted(by.items())})


def paired(A, B, key="ok"):
    qa, qb = collections.defaultdict(list), collections.defaultdict(list)
    for r in A:
        qa[r["uid"]].append(r[key])
    for r in B:
        qb[r["uid"]].append(r[key])
    U = sorted(set(qa) & set(qb))
    diffs = [sum(qa[u]) / len(qa[u]) - sum(qb[u]) / len(qb[u]) for u in U]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(diffs) for _ in U) / len(U) for _ in range(2000))
    return dict(n=len(U), mean=sum(diffs) / len(U), lo=bs[50], hi=bs[1949])


def pair(A, B):
    return dict(d_acc=paired(A, B),
                d_crops=paired([dict(r, ok=r["n_crops"]) for r in A], [dict(r, ok=r["n_crops"]) for r in B]))


out = {"E1": {}, "E2": {}, "train": {}, "logs": {}}
for st in (80, 160):
    for e, sfx in (("E1", ""), ("E2", "_perturb")):
        fc, zero, r1 = load(fd, f"fc{st}{sfx}"), load(zd, f"zero{st}{sfx}"), load(rd, f"r1_{st}{sfx}")
        if not (fc and zero):
            continue
        cell = dict(fc=summ(fc), zero=summ(zero), fc_vs_zero=pair(fc, zero))
        if r1:
            cell.update(r1=summ(r1), zero_vs_r1=pair(zero, r1))
        out[e][st] = cell

p = out["E1"].get(160, {}).get("fc_vs_zero")
if p:
    acc, cr = p["d_acc"], p["d_crops"]
    crops_excl = cr["lo"] > 0 or cr["hi"] < 0
    acc_incl = acc["lo"] <= 0 <= acc["hi"]
    if acc["lo"] > 0 or (crops_excl and acc["lo"] >= -0.03):
        verdict = "H"
    elif acc_incl and not crops_excl:
        verdict = "rival"
    else:
        verdict = "undecided"
    out["verdict"] = dict(rule="E1 step 160 fc - zero", verdict=verdict, d_acc=acc, d_crops=cr)
else:
    out["verdict"] = dict(verdict="not_testable", reason="E1 step-160 evals missing")

for a, name in (("zero", "grpo_v3_A_fc_zero"), ("fc", "grpo_v3_A_fc_fc")):
    try:
        Rr = [json.loads(line) for line in open(f"{outs}/{name}_rollouts.jsonl")]
    except FileNotFoundError:
        continue
    w = collections.defaultdict(list)
    for r in Rr:
        w[(r["step"] - 1) // 40].append(r)
    out["train"][a] = {f"s{40*k+1}-{40*k+40}": dict(n=len(v), direct=sum(r["crops"] == 0 for r in v) / len(v),
                                                    crops=sum(r["crops"] for r in v) / len(v),
                                                    acc=sum(float(r["correct"]) for r in v) / len(v))
                       for k, v in sorted(w.items())}
    vals = collections.defaultdict(lambda: collections.defaultdict(list))
    step = 0
    for line in open(f"{outs}/{name}.log", errors="ignore"):
        if "time_per_step" in line:
            step += 1
        if "[fc_forks]" not in line and "[fc_credit]" not in line:
            continue
        win = f"s{40*(step//40)+1}-{40*(step//40)+40}"
        for key in ("forks", "truncated", "decision_states", "mean_v", "crop_turns", "credited", "WR", "RR", "WW", "RW",
                    "mean_c", "mean_A_replaced"):
            m = re.search(rf"\b{key}=(-?[0-9.]+)", line)
            if m:
                vals[win][key].append(float(m.group(1)))
    out["logs"][a] = {k: {kk: dict(n=len(v), mean=sum(v) / len(v)) for kk, v in d.items()} for k, d in sorted(vals.items())}
    ts = [float(m.group(1)) for m in re.finditer(r"time_per_step: ([0-9.]+)", open(f"{outs}/{name}.log", errors="ignore").read())]
    out["logs"][a]["time_per_step"] = dict(n=len(ts), mean=sum(ts) / max(1, len(ts)))
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps(out["verdict"], indent=1))
