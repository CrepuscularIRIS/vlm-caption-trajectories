"""A4 follow-up: first-order pressure on the number of crops among trajectories that cropped at least once (later
crops only), per 40-step block. push = mean over groups of sum_i A_i (n_i - mean n) / |subset|, A = RLOO advantage of
r = Y(1 - lam N). <0: the update favours fewer crops on that question. Also accuracy by crop count (pooled).
Usage: python3 push_later.py <rollouts.jsonl> <lam> <out.json>
"""
import collections
import json
import statistics as st
import sys


def main() -> None:
    path, lam, dest = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    g = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        g[(int(r["step"]), r["uid"])].append((float(r["correct"]), int(r["crops"])))
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for (s, _), x in g.items():
        y = [a[0] for a in x]
        n = [a[1] for a in x]
        R = [yi * (1 - lam * ni) for yi, ni in zip(y, n)]
        tot = sum(R)
        A = [ri - (tot - ri) / (len(R) - 1) for ri in R]
        b = f"{(s - 1) // 40 * 40 + 1}-{min((s - 1) // 40 * 40 + 40, 165)}"
        out[b]["acc_by_n"].extend((min(ni, 4), yi) for ni, yi in zip(n, y))
        sub = [i for i in range(len(x)) if n[i] >= 1]
        if len(sub) < 2:
            continue
        nb = st.fmean(n[i] for i in sub)
        out[b]["push_crop1plus"].append(sum(A[i] * (n[i] - nb) for i in sub) / len(sub))
        out[b]["push_crop1plus_noncap"].append(sum(A[i] * (n[i] - nb) for i in sub if n[i] < 7) / len(sub))
    res = {}
    for b, d in sorted(out.items(), key=lambda kv: int(kv[0].split("-")[0])):
        acc = collections.defaultdict(list)
        for k, v in d["acc_by_n"]:
            acc[k].append(v)
        res[b] = {k: {"mean": st.fmean(v), "se": st.stdev(v) / len(v) ** 0.5, "groups": len(v)}
                  for k, v in d.items() if k != "acc_by_n"}
        res[b]["acc_by_crops(4=4+)"] = {str(k): {"acc": st.fmean(v), "n": len(v)} for k, v in sorted(acc.items())}
    json.dump(res, open(dest, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
