"""E[N] = P(N>=1) + E[(N-1)+] for every existing E1/E2 eval (Pro B §2.7): is a lower crop count coming from fewer first
crops or from fewer later crops (the "2 steps became 4-5" part)? Also the long tail P(N>=3), P(N>=5), and paired
differences vs a reference run (question-level bootstrap, 2,000 resamples, seed 0).
Usage: python3 later_decomp.py <out.json> <ref_name> name=path [name=path ...]
"""
import collections
import json
import random
import statistics as st
import sys


def load(path: str) -> dict:
    q = collections.defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            q[r["uid"]].append((int(r["n_crops"]), int(bool(r["ok"]))))
    return q


def per_q(q: dict) -> dict:
    return {u: {"acc": st.fmean(y for _, y in v), "crops": st.fmean(n for n, _ in v),
                "first": st.fmean(n >= 1 for n, _ in v), "later": st.fmean(max(n - 1, 0) for n, _ in v),
                "ge3": st.fmean(n >= 3 for n, _ in v)} for u, v in q.items()}


def summary(q: dict) -> dict:
    s = [x for v in q.values() for x in v]
    return {"samples": len(s), "acc": st.fmean(y for _, y in s), "crops": st.fmean(n for n, _ in s),
            "P(N>=1)": st.fmean(n >= 1 for n, _ in s), "later_per_q E[(N-1)+]": st.fmean(max(n - 1, 0) for n, _ in s),
            "P(N>=3)": st.fmean(n >= 3 for n, _ in s), "P(N>=5)": st.fmean(n >= 5 for n, _ in s),
            "acc_given_N>=3": st.fmean([y for n, y in s if n >= 3] or [float("nan")])}


def paired(a: dict, b: dict) -> dict:
    us = sorted(set(a) & set(b))
    rng = random.Random(0)
    out = {}
    for k in ("acc", "crops", "first", "later", "ge3"):
        d = [a[u][k] - b[u][k] for u in us]
        bs = sorted(st.fmean(d[rng.randrange(len(d))] for _ in d) for _ in range(2000))
        out[k] = [st.fmean(d), bs[50], bs[1949]]
    out["questions"] = len(us)
    return out


def main() -> None:
    dest, ref = sys.argv[1], sys.argv[2]
    runs = dict(a.split("=", 1) for a in sys.argv[3:])
    data = {k: load(p) for k, p in runs.items()}
    res = {"summary": {k: summary(v) for k, v in data.items()}, "ref": ref,
           "paired_vs_ref(mean, lo95, hi95)": {k: paired(per_q(v), per_q(data[ref])) for k, v in data.items() if k != ref}}
    json.dump(res, open(dest, "w"), indent=1)
    for k, s in res["summary"].items():
        p = res["paired_vs_ref(mean, lo95, hi95)"].get(k)
        extra = "" if not p else "  vs %s: acc %+.3f [%+.3f,%+.3f] first %+.3f later %+.3f [%+.3f,%+.3f]" % (
            ref, *p["acc"], p["first"][0], *p["later"])
        print("%-10s acc %.3f crops %.2f P1 %.3f later %.2f P>=3 %.3f P>=5 %.3f%s" % (
            k, s["acc"], s["crops"], s["P(N>=1)"], s["later_per_q E[(N-1)+]"], s["P(N>=3)"], s["P(N>=5)"], extra))


if __name__ == "__main__":
    main()
