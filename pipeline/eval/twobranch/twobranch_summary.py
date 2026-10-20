"""Per-question two-branch log + pre-registered test (thresholds fixed before the run, 2026-10-19).

Inputs: traj_A_sft_direct.jsonl, traj_A_sft_crop.jsonl (eval_twobranch.py, 4 samples per uid per branch).
Per uid: acc_direct, acc_crop (k/4), dV = acc_crop - acc_direct, mean crops (crop branch), mean generated tokens per
branch, first-turn rejection counts, category:
  direct_ok   acc_direct >= 3/4                      (direct answer can solve it)
  needs_look  acc_crop >= 3/4 and acc_direct <= 1/4  (only the extra look solves it)
  both_wrong  acc_direct <= 1/4 and acc_crop <= 1/4
  mixed       otherwise
Test (H vs rival "4-sample noise"):
  S1 = #uids with |dV| >= 3/4; null: within each uid pool the 8 outcomes and reassign 4/4 to branches at random
       (10,000 permutations); H predicts S1 > null 95th percentile, rival predicts S1 within the null.
  S2 = split-half sign agreement: among uids where dV on samples {0,1} and on {2,3} are both nonzero, share with the
       same sign; same permutation null; H predicts S2 > null 95th percentile.
Writes per_question.csv (no question text) and test.json.
"""
import collections, csv, json, random, sys

d = sys.argv[1]
def load(branch):
    by = collections.defaultdict(dict)
    for l in open(f"{d}/traj_A_sft_{branch}.jsonl"):
        r = json.loads(l); by[r["uid"]][r["sample_idx"]] = r
    return by
D, C = load("direct"), load("crop")
uids = sorted(set(D) & set(C))
rows, pairs = [], []
for u in uids:
    dd, cc = [D[u][i] for i in sorted(D[u])], [C[u][i] for i in sorted(C[u])]
    od, oc = [r["ok"] for r in dd], [r["ok"] for r in cc]
    ad, ac = sum(od) / len(od), sum(oc) / len(oc)
    cat = ("direct_ok" if ad >= .75 else "needs_look" if ac >= .75 and ad <= .25
           else "both_wrong" if ad <= .25 and ac <= .25 else "mixed")
    rows.append(dict(uid=u, source=dd[0]["source"], stratum=dd[0]["layer"], gt=dd[0]["gt"], n=len(od),
                     acc_direct=ad, acc_crop=ac, dV=ac - ad, category=cat,
                     crops_crop=sum(r["n_crops"] for r in cc) / len(cc),
                     tok_direct=sum(r["gen_tokens"] for r in dd) / len(dd), tok_crop=sum(r["gen_tokens"] for r in cc) / len(cc),
                     rej_direct=sum(r.get("first_tries", 0) for r in dd), rej_crop=sum(r.get("first_tries", 0) for r in cc),
                     unmet_direct=sum(bool(r.get("error")) for r in dd), unmet_crop=sum(bool(r.get("error")) for r in cc)))
    pairs.append((od, oc))
with open(f"{d}/per_question.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)

def stats(P):
    s1 = sum(abs(sum(c) / 4 - sum(o) / 4) >= .75 for o, c in P)
    agree = tot = 0
    for o, c in P:
        a1, a2 = (c[0] + c[1]) - (o[0] + o[1]), (c[2] + c[3]) - (o[2] + o[3])
        if a1 and a2:
            tot += 1; agree += (a1 > 0) == (a2 > 0)
    return s1, (agree / tot if tot else float("nan")), tot
s1, s2, n2 = stats(pairs)
rng = random.Random(20261019); null1, null2 = [], []
for _ in range(10000):
    P = []
    for o, c in pairs:
        x = o + c; rng.shuffle(x); P.append((x[:4], x[4:]))
    a, b, _ = stats(P); null1.append(a); null2.append(b)
null1.sort(); null2 = sorted(v for v in null2 if v == v)
q95 = lambda v: v[int(.95 * len(v))]
cats = collections.Counter(r["category"] for r in rows)
out = dict(n_uids=len(uids), categories=dict(cats),
           acc_direct=sum(r["acc_direct"] for r in rows) / len(rows), acc_crop=sum(r["acc_crop"] for r in rows) / len(rows),
           crops_crop=sum(r["crops_crop"] for r in rows) / len(rows),
           tok_direct=sum(r["tok_direct"] for r in rows) / len(rows), tok_crop=sum(r["tok_crop"] for r in rows) / len(rows),
           unmet=dict(direct=sum(r["unmet_direct"] for r in rows), crop=sum(r["unmet_crop"] for r in rows)),
           S1=s1, S1_null95=q95(null1), S1_null_mean=sum(null1) / len(null1),
           S2=s2, S2_n=n2, S2_null95=q95(null2), S2_null_mean=sum(null2) / len(null2),
           H_S1=s1 > q95(null1), H_S2=s2 > q95(null2))
json.dump(out, open(f"{d}/test.json", "w"), indent=1)
print(json.dumps(out, indent=1))
