"""Stage A zero-GPU ledger diagnostics (corpus ROADMAP §1 A1/A2/A4, adapted to later crops). Reads one RLOO training
rollout ledger (one row per sampled trajectory: step, uid, crops, correct, score, gen_tokens). Prints aggregate stats
only; no question text leaves this script.

A1: share of the lambda term in sum|A| (and token-weighted), by group type (all-correct / mixed / all-wrong).
A2: support among correct samples: share with 0 crops, 1 crop, >=2 crops; share of groups whose correct set holds
    both a 1-crop and a >=2-crop sample (the only groups where a later-crop cost can compare like with like).
A4 push: per group, sum_i A_i (n_i - mean n) (unweighted, and weighted by generated tokens as in EasyR1's token-mean
    loss): >0 means the update favours more crops on that question, to first order.
A4: within-group implicit bonus for later crops: mean Y(>=2 crops) - mean Y(1 crop) in groups with both, against the
    explicit cost; and mean RLOO advantage of correct samples by crop count.
Usage: python3 ledger_diag.py <rollouts.jsonl> <lam> <out.json>
"""
import collections
import json
import statistics
import sys

BLOCKS = [(1, 10), (11, 40), (41, 80), (81, 120), (121, 999)]


def block(step: int) -> str:
    for a, b in BLOCKS:
        if a <= step <= b:
            return f"{a}-{min(b, 165)}"
    raise ValueError(step)


def loo(xs: list) -> list:
    n, s = len(xs), sum(xs)
    return [x - (s - x) / (n - 1) for x in xs]


def main() -> None:
    path, lam, dest = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    groups = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        groups[(int(r["step"]), r["uid"])].append(
            (float(r["correct"]), int(r["crops"]), int(r["gen_tokens"]), float(r["score"]), r["terminal"]))
    out = collections.defaultdict(lambda: collections.defaultdict(float))
    lists = collections.defaultdict(lambda: collections.defaultdict(list))
    score_check = []
    for (step, _), g in groups.items():
        if len(g) < 2:
            continue
        b = block(step)
        y = [s[0] for s in g]
        n = [s[1] for s in g]
        tok = [s[2] for s in g]
        r = [yi * (1 - lam * ni) for yi, ni in zip(y, n)]
        score_check += [abs(ri - s[3]) for ri, s in zip(r, g)]
        a = loo(r)
        a_cost = [-lam * v for v in loo([yi * ni for yi, ni in zip(y, n)])]
        kind = "all_correct" if min(y) == 1 else "all_wrong" if max(y) == 0 else "mixed"
        o = out[b]
        o[f"groups_{kind}"] += 1
        o["groups"] += 1
        o[f"absA_{kind}"] += sum(map(abs, a))
        o[f"absAcost_{kind}"] += sum(map(abs, a_cost))
        o[f"absA_tok_{kind}"] += sum(abs(x) * t for x, t in zip(a, tok))
        o[f"absAcost_tok_{kind}"] += sum(abs(x) * t for x, t in zip(a_cost, tok))
        o["samples"] += len(g)
        o["cap_hits"] += sum(s[4] != "answer" or s[1] >= 7 for s in g)
        nbar = statistics.fmean(n)
        lists[b]["push_unweighted"].append(sum(ai * (ni - nbar) for ai, ni in zip(a, n)) / len(g))
        lists[b]["push_tokw"].append(sum(ai * t * (ni - nbar) for ai, ni, t in zip(a, n, tok)) / sum(tok))
        lists[b]["push_tokw_cost_only"].append(sum(ai * t * (ni - nbar) for ai, ni, t in zip(a_cost, n, tok)) / sum(tok))
        cor = [(ni, ai) for yi, ni, ai in zip(y, n, a) if yi == 1]
        o["correct"] += len(cor)
        o["correct_0crop"] += sum(ni == 0 for ni, _ in cor)
        o["correct_1crop"] += sum(ni == 1 for ni, _ in cor)
        o["correct_2pluscrop"] += sum(ni >= 2 for ni, _ in cor)
        if any(ni == 1 for ni, _ in cor) and any(ni >= 2 for ni, _ in cor):
            o["groups_correct_has_1_and_2plus"] += 1
        for ni, ai in cor:
            lists[b][f"adv_correct_n{min(ni, 3)}"].append(ai)
        one = [yi for yi, ni in zip(y, n) if ni == 1]
        more = [yi for yi, ni in zip(y, n) if ni >= 2]
        if one and more:
            lists[b]["dY_2plus_minus_1"].append(statistics.fmean(more) - statistics.fmean(one))
            lists[b]["dN_2plus_minus_1"].append(statistics.fmean([ni for ni in n if ni >= 2]) - 1)
    res = {"score_formula_max_abs_err": max(score_check), "lambda": lam, "blocks": {}}
    for b in sorted(out, key=lambda k: int(k.split("-")[0])):
        o, L = out[b], lists[b]
        d = {"groups": o["groups"], "samples": o["samples"]}
        for k in ("all_correct", "mixed", "all_wrong"):
            d[f"share_groups_{k}"] = o[f"groups_{k}"] / o["groups"]
            tot = sum(o[f"absA_{j}"] for j in ("all_correct", "mixed", "all_wrong"))
            d[f"A1_cost_share_of_absA_{k}"] = o[f"absAcost_{k}"] / o[f"absA_{k}"] if o[f"absA_{k}"] else None
            d[f"A1_cost_share_tok_{k}"] = o[f"absAcost_tok_{k}"] / o[f"absA_tok_{k}"] if o[f"absA_tok_{k}"] else None
            d[f"share_total_absA_from_{k}"] = o[f"absA_{k}"] / tot if tot else None
        d["A2_correct_share_0crop"] = o["correct_0crop"] / o["correct"]
        d["A2_correct_share_1crop"] = o["correct_1crop"] / o["correct"]
        d["A2_correct_share_2plus"] = o["correct_2pluscrop"] / o["correct"]
        d["A2_groups_correct_has_1_and_2plus"] = o["groups_correct_has_1_and_2plus"] / o["groups"]
        d["cap_or_nonanswer_share"] = o["cap_hits"] / o["samples"]
        dy = L["dY_2plus_minus_1"]
        d["A4_groups_with_1_and_2plus"] = len(dy)
        if dy:
            d["A4_mean_dY_2plus_minus_1"] = statistics.fmean(dy)
            d["A4_dY_se"] = statistics.stdev(dy) / len(dy) ** 0.5 if len(dy) > 1 else None
            d["A4_mean_extra_crops"] = statistics.fmean(L["dN_2plus_minus_1"])
            d["A4_explicit_cost_of_extra_crops"] = lam * d["A4_mean_extra_crops"]
        for k in ("push_unweighted", "push_tokw", "push_tokw_cost_only"):
            d[f"A4_{k}_mean"] = statistics.fmean(L[k])
            d[f"A4_{k}_se"] = statistics.stdev(L[k]) / len(L[k]) ** 0.5
        for k in sorted(L):
            if k.startswith("adv_correct"):
                d[f"A4_{k}_mean"] = statistics.fmean(L[k])
                d[f"A4_{k}_n"] = len(L[k])
        res["blocks"][b] = d
    json.dump(res, open(dest, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
