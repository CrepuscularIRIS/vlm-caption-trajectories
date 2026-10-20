"""dV table for the first-turn GRPO term (user rule 2026-10-19: branch value = mean over met samples; dV = 0 when
either branch has < 2 met samples). Inputs: the 300-question run (evidence/twobranch_20261019) and the 1,700-question
run (evidence/twobranch_pool_20261020), same checkpoint/script/seeds. Writes dv_table.json {uid: dV} and
dv_summary.json with the pre-registered statistics:
  share_big = share of the 2,000 questions with |dV| >= 0.5; share_pos / share_neg = shares with dV >= 0.5 / <= -0.5;
  share_zero_rule = share set to 0 by the < 2 met rule.
"""
import collections, json, sys
dirs, out = sys.argv[1:-1], sys.argv[-1]
S = {b: collections.defaultdict(dict) for b in ("direct", "crop")}
for d in dirs:
    for b in S:
        for l in open(f"{d}/traj_A_sft_{b}.jsonl"):
            r = json.loads(l); S[b][r["uid"]][r["sample_idx"]] = r
uids = sorted(set(S["direct"]) | set(S["crop"]))
dv, src, zero = {}, {}, 0
for u in uids:
    md = [r["ok"] for r in S["direct"].get(u, {}).values() if not r.get("error")]
    mc = [r["ok"] for r in S["crop"].get(u, {}).values() if not r.get("error")]
    if len(md) < 2 or len(mc) < 2:
        dv[u] = 0.0; zero += 1
    else:
        dv[u] = sum(mc) / len(mc) - sum(md) / len(md)
    r0 = next(iter((S["direct"].get(u) or S["crop"][u]).values())); src[u] = r0["source"]
n = len(uids)
summ = dict(n=n, share_big=sum(abs(v) >= .5 for v in dv.values()) / n, share_pos=sum(v >= .5 for v in dv.values()) / n,
            share_neg=sum(v <= -.5 for v in dv.values()) / n, share_zero_rule=zero / n,
            mean_dV=sum(dv.values()) / n,
            by_source={s: dict(n=sum(src[u] == s for u in uids), share_big=sum(abs(dv[u]) >= .5 for u in uids if src[u] == s) / max(1, sum(src[u] == s for u in uids)),
                            mean_dV=sum(dv[u] for u in uids if src[u] == s) / max(1, sum(src[u] == s for u in uids))) for s in sorted(set(src.values()))},
            H=None)
summ["H"] = summ["share_big"] >= .10 and summ["share_pos"] > 0 and summ["share_neg"] > 0
json.dump(dv, open(f"{out}/dv_table.json", "w")); json.dump(summ, open(f"{out}/dv_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
