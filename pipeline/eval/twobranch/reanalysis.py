"""Committed re-analysis of the t-twobranch run (no new samples): first_action_unmet samples are EXCLUDED instead of
scored wrong (interpret finding f-unmet-scoring). Rule fixed before running (the host had seen an informal recount with
the same rule): a question is kept if each branch has >= 1 met sample; accuracy per branch = mean over met samples.
Categories as before (direct_ok acc_d>=3/4; needs_look acc_c>=3/4 & acc_d<=1/4; both_wrong both<=1/4; mixed).
Unmet counts are kept as separate columns (a forced-preference signal), and questions with >=2/4 unmet in a branch are
flagged pref_direct / pref_crop. S1 = #|dV|>=3/4, S2 = split-half sign agreement (halves = sample_idx {0,1} vs {2,3},
met samples only, both halves non-empty in both branches); null = within-question shuffle of met outcomes between
branches keeping each branch's met count (10,000 permutations, seed 20261019).
Also writes inspection_sample.csv: 10 uids per category (seed 20261019) for the mentor's hand inspection.
"""
import collections, csv, json, random, sys
d = sys.argv[1]
def load(b):
    by = collections.defaultdict(dict)
    for l in open(f"{d}/traj_A_sft_{b}.jsonl"):
        r = json.loads(l); by[r["uid"]][r["sample_idx"]] = r
    return by
D, C = load("direct"), load("crop")
rows, pairs, excl = [], [], []
for u in sorted(set(D) & set(C)):
    md = {i: r["ok"] for i, r in D[u].items() if not r.get("error")}
    mc = {i: r["ok"] for i, r in C[u].items() if not r.get("error")}
    ud, uc = len(D[u]) - len(md), len(C[u]) - len(mc)
    r0 = next(iter(D[u].values()))
    if not md or not mc:
        excl.append(dict(uid=u, source=r0["source"], unmet_direct=ud, unmet_crop=uc)); continue
    ad, ac = sum(md.values()) / len(md), sum(mc.values()) / len(mc)
    cat = ("direct_ok" if ad >= .75 else "needs_look" if ac >= .75 and ad <= .25
           else "both_wrong" if ad <= .25 and ac <= .25 else "mixed")
    metc = [r for r in C[u].values() if not r.get("error")]
    rows.append(dict(uid=u, source=r0["source"], stratum=r0["layer"], gt=r0["gt"], met_direct=len(md), met_crop=len(mc),
                     unmet_direct=ud, unmet_crop=uc, pref_direct=int(uc >= 2), pref_crop=int(ud >= 2),
                     acc_direct=ad, acc_crop=ac, dV=ac - ad, category=cat,
                     crops_crop=sum(r["n_crops"] for r in metc) / len(metc),
                     tok_direct=sum(r["gen_tokens"] for r in D[u].values() if not r.get("error")) / len(md),
                     tok_crop=sum(r["gen_tokens"] for r in metc) / len(metc)))
    pairs.append((md, mc))
with open(f"{d}/per_question_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
def stats(P):
    s1 = sum(abs(sum(c.values()) / len(c) - sum(o.values()) / len(o)) >= .75 for o, c in P)
    ag = tot = 0
    for o, c in P:
        h = [[o.get(i) for i in (0, 1) if i in o], [o.get(i) for i in (2, 3) if i in o],
             [c.get(i) for i in (0, 1) if i in c], [c.get(i) for i in (2, 3) if i in c]]
        if not all(h): continue
        a1 = sum(h[2]) / len(h[2]) - sum(h[0]) / len(h[0]); a2 = sum(h[3]) / len(h[3]) - sum(h[1]) / len(h[1])
        if a1 and a2: tot += 1; ag += (a1 > 0) == (a2 > 0)
    return s1, (ag / tot if tot else float("nan")), tot
s1, s2, n2 = stats(pairs)
rng = random.Random(20261019); n1, n2l = [], []
for _ in range(10000):
    P = []
    for o, c in pairs:
        keys = [("o", i) for i in o] + [("c", i) for i in c]; vals = list(o.values()) + list(c.values()); rng.shuffle(vals)
        no, nc = {}, {}
        for (b, i), v in zip(keys, vals): (no if b == "o" else nc)[i] = v
        P.append((no, nc))
    a, b, _ = stats(P); n1.append(a); n2l.append(b)
n1.sort(); n2l = sorted(v for v in n2l if v == v); q = lambda v: v[int(.95 * len(v))]
cats = collections.Counter(r["category"] for r in rows)
out = dict(n_kept=len(rows), n_excluded=len(excl), excluded=excl, categories=dict(cats),
           by_source={s: dict(collections.Counter(r["category"] for r in rows if r["source"] == s)) for s in sorted({r["source"] for r in rows})},
           pref_direct=sum(r["pref_direct"] for r in rows), pref_crop=sum(r["pref_crop"] for r in rows),
           acc_direct_met=sum(r["acc_direct"] for r in rows) / len(rows), acc_crop_met=sum(r["acc_crop"] for r in rows) / len(rows),
           S1=s1, S1_null95=q(n1), S1_null_mean=sum(n1) / len(n1), S2=s2, S2_n=n2, S2_null95=q(n2l), S2_null_mean=sum(n2l) / len(n2l),
           H_S1=s1 > q(n1), H_S2=s2 > q(n2l))
json.dump(out, open(f"{d}/test_v2.json", "w"), indent=1)
rng = random.Random(20261019); samp = []
for c in ["direct_ok", "needs_look", "both_wrong", "mixed"]:
    pool = [r for r in rows if r["category"] == c]
    for r in rng.sample(pool, min(10, len(pool))): samp.append(dict(uid=r["uid"], category=c, source=r["source"], acc_direct=r["acc_direct"], acc_crop=r["acc_crop"], crops_crop=r["crops_crop"]))
with open(f"{d}/inspection_sample.csv", "w", newline="") as f:
    w = csv.DictWriter(f, samp[0].keys()); w.writeheader(); w.writerows(samp)
print(json.dumps({k: v for k, v in out.items() if k != "excluded"}, indent=1))
