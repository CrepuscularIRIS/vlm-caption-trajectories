"""Summary of the segment-value MVE: per case and prefix V = mean Y (and mean R = Y(1-.01N)), plus over the 9 core
cases (paired by case; bootstrap 2,000, seed 0): S1 = mean[V(P1)-V(P0)], S2 = mean[V(P1t)-V(P1m)] over cases with a P1m, S3 = mean V(P2t);
the 4 mixed cases separately; per generating policy descriptive. Usage: python summarize_segment.py <dir>"""
import collections, json, random, sys
from pathlib import Path
d = Path(sys.argv[1])
R = [json.loads(l) for p in sorted(d.glob("runs_*.jsonl")) for l in open(p)]
V = collections.defaultdict(lambda: collections.defaultdict(list))
meta = {}
for r in R:
    V[r["id"]][r["prefix"]].append(r)
    meta[r["id"]] = (r["group"], r["policy"])
per = {cid: {p: dict(n=len(v), V=sum(x["y"] for x in v) / len(v), R=sum(x["r"] for x in v) / len(v),
                     crops=sum(x["crops"] for x in v) / len(v), tok=sum(x["gen_tokens"] for x in v) / len(v))
             for p, v in ps.items()} | {"group": meta[cid][0], "policy": meta[cid][1]} for cid, ps in V.items()}
def stat(ids, f):
    xs = [f(per[i]) for i in ids if f(per[i]) is not None]
    if not xs: return None
    rng = random.Random(0); bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(2000))
    return dict(n=len(xs), mean=sum(xs) / len(xs), lo=bs[50], hi=bs[1949])
g = lambda p, k: (lambda c: c[p]["V"] if p in c else None)
S = {}
for grp in ("core", "mixed"):
    ids = [i for i in per if per[i]["group"] == grp]
    S[grp] = dict(S1=stat(ids, lambda c: c["P1t"]["V"] - c["P0"]["V"]), S2=stat(ids, lambda c: (c["P1t"]["V"] - c["P1m"]["V"]) if "P1m" in c else None),
                  S3=stat(ids, lambda c: c["P2t"]["V"] if "P2t" in c else None),
                  V0=stat(ids, lambda c: c["P0"]["V"]), V1=stat(ids, lambda c: c["P1t"]["V"]))
json.dump(dict(per_case=per, summary=S, n_runs=len(R)), open(d / "summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
