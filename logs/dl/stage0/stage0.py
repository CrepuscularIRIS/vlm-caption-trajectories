"""Stage 0 (Pro/Codex redesign READOUT 10-23): is the per-state branch gap D signal or noise, and how much routing
room is there, using only forks already collected. CPU only; no model, no new samples.

Inputs (raw/): audit_states.json (64 reference states, r1@40, 4+4 forks each), score_ref.json (reference tag scores),
grpo_v3_A_dl_main_forks.jsonl.* (240 training states, evolving main policy, 4+4 forks each).
Primary split fixed in advance: forks k in {0,1} vs {2,3}; the other two 2+2 pairings are reported as a check.
Usage: python3 stage0.py  (writes stage0.json next to this file)
"""
import glob
import hashlib
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw"
SPLITS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
B, SEED = 10000, 0


def q(forks, action, ks, key="r"):
    xs = [f[key] for f in forks if (f["action"] == "answer") == (action == "answer") and int(f["k"]) in ks]
    return sum(xs) / len(xs)


def gap(forks, ks, key="r"):
    return q(forks, "answer", ks, key) - q(forks, "crop", ks, key)


def pearson(x, y):
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else None


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2
        i = j + 1
    return r


def boot(fn, n):
    rng = random.Random(SEED)
    vals = sorted(fn([rng.randrange(n) for _ in range(n)]) for _ in range(B))
    vals = [v for v in vals if v is not None]
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def reliability(states):
    """Split-half agreement of D. cov(D_h1, D_h2) estimates the variance of the true gap (independent fork noise);
    divided by var(D_full) it is the share of the 4+4 estimate's variance that is signal."""
    out = {}
    for h1, h2 in SPLITS:
        d1 = [gap(s, h1) for s in states]
        d2 = [gap(s, h2) for s in states]
        df = [gap(s, (0, 1, 2, 3)) for s in states]
        cov = statistics.covariance(d1, d2)
        both = [(a, b) for a, b in zip(d1, d2) if a != 0 and b != 0]
        r = pearson(d1, d2)
        out[f"{h1}|{h2}"] = {
            "pearson": r, "spearman": pearson(ranks(d1), ranks(d2)),
            "spearman_brown_4": 2 * r / (1 + r) if r is not None else None,
            "signal_var_est": cov, "var_D_full": statistics.variance(df),
            "signal_share_of_D_full": cov / statistics.variance(df),
            "sign_agree_both_nonzero": [sum((a > 0) == (b > 0) for a, b in both), len(both)],
            "pearson_ci95": boot(lambda idx: pearson([d1[i] for i in idx], [d2[i] for i in idx]), len(d1)),
        }
    return out


def decompose(states):
    """D = dY - 0.01 * d(Y*N): accuracy part vs crop-cost part, and where answering costs accuracy."""
    ks = (0, 1, 2, 3)
    d = [gap(s, ks) for s in states]
    dy = [gap(s, ks, "y") for s in states]
    cost = [a - b for a, b in zip(d, dy)]
    return {
        "mean_D": statistics.fmean(d), "mean_dY": statistics.fmean(dy), "mean_cost_part": statistics.fmean(cost),
        "D_pos": sum(x > 0 for x in d), "D_neg": sum(x < 0 for x in d), "D_zero": sum(x == 0 for x in d),
        "D_pos_with_dY_pos": sum(a > 0 and b > 0 for a, b in zip(d, dy)),
        "D_pos_cost_only": sum(a > 0 and b == 0 for a, b in zip(d, dy)),
        "D_pos_with_dY_neg": sum(a > 0 and b < 0 for a, b in zip(d, dy)),
        "mean_acc_answer": statistics.fmean(q(s, "answer", ks, "y") for s in states),
        "mean_acc_crop": statistics.fmean(q(s, "crop", ks, "y") for s in states),
        "mean_crops_crop_branch": statistics.fmean(q(s, "crop", ks, "n_crops") for s in states),
    }


def best_threshold(score, d):
    """Answer iff score > t; t over all cut points (incl. all-answer / all-crop). Returns (t, in-sample gain vs all-crop)."""
    cuts = sorted(set(score))
    cands = [min(cuts) - 1] + cuts
    best = max(cands, key=lambda t: sum(x for s, x in zip(score, d) if s > t))
    return best, sum(x for s, x in zip(score, d) if s > best) / len(d)


def room(states, z):
    """Routing value relative to always-crop (mean of chosen-arm Q minus Q_crop), on the 64 audit states.
    in-sample: Codex's same-table numbers. cross-fit: fit on half 1, score on half 2, and the reverse, averaged."""
    full = [gap(s, (0, 1, 2, 3)) for s in states]
    n = len(full)
    t, thr_in = best_threshold(z, full)
    oracle_in = sum(max(x, 0) for x in full) / n
    res = {"in_sample": {"oracle": oracle_in, "best_threshold": thr_in, "threshold": t,
                         "oracle_minus_threshold": oracle_in - thr_in,
                         "always_answer": statistics.fmean(full)}}
    folds = []
    for h1, h2 in SPLITS[:1]:
        for fit, ev in ((h1, h2), (h2, h1)):
            dfit = [gap(s, fit) for s in states]
            dev = [gap(s, ev) for s in states]
            tt, _ = best_threshold(z, dfit)
            folds.append({"oracle_from_fit": [x if a > 0 else 0.0 for a, x in zip(dfit, dev)],
                          "threshold": [x if s > tt else 0.0 for s, x in zip(z, dev)],
                          "always_answer": dev, "t": tt})
    per = {k: [statistics.fmean(f[k][i] for f in folds) for i in range(n)]
           for k in ("oracle_from_fit", "threshold", "always_answer")}
    diff = [a - b for a, b in zip(per["oracle_from_fit"], per["threshold"])]
    res["cross_fit"] = {k: statistics.fmean(v) for k, v in per.items()}
    res["cross_fit"]["oracle_minus_threshold"] = {"mean": statistics.fmean(diff),
                                                  "ci95": boot(lambda idx: statistics.fmean(diff[i] for i in idx), n)}
    res["cross_fit"]["thresholds"] = [f["t"] for f in folds]
    res["takeover_10pct_scale"] = 0.1
    return res


def main() -> None:
    audit = json.load(open(RAW / "audit_states.json"))["states"]
    ref = {s["uid"]: s for s in json.load(open(RAW / "score_ref.json"))["scores"]}
    audit = [s for s in audit if s["usable"] and len(s["forks"]) == 8]
    a_forks = [s["forks"] for s in audit]
    z = [ref[s["uid"]]["z"] for s in audit]

    rows = [json.loads(line) for p in sorted(glob.glob(str(RAW / "grpo_v3_A_dl_main_forks.jsonl.*")))
            for line in open(p) if line.strip()]
    groups = {}
    for r in rows:
        if "action" in r:
            groups.setdefault((r["pid"], r["call"], r["owner_row"]), []).append(r)
    t_forks = [v for v in groups.values() if len(v) == 8]

    out = {
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.iterdir())},
        "inputs": {"audit_states": len(audit), "train_states": len(t_forks),
                   "audit_invalid_forks": sum(f["terminal"] == "invalid" for s in a_forks for f in s)},
        "reliability": {"audit_ref": reliability(a_forks), "train_main": reliability(t_forks)},
        "decompose": {"audit_ref": decompose(a_forks), "train_main": decompose(t_forks)},
        "room_audit_ref": room(a_forks, z),
    }
    (HERE / "stage0.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
