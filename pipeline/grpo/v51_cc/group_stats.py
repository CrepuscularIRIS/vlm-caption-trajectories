"""Rollout group census: all-correct / all-wrong / mixed groups, split by direct vs crop.

Advantage is recomputed from the logged per-rollout score exactly as each run's trainer did:
  outcome = (score - mean) / (std_unbiased + 1e-6)          (cc, cc_dsasym, cropcost)
  outcome = (score - mean) / max(std_unbiased, 0.25)        (floor_*; CAPTION_STD_FLOOR)
  cc runs add price channel: (price - group mean price), 0 if the group has no correct rollout.
Supervised tokens = gen_tokens (response_mask sum). "Effective" tokens = those with |adv| > 1e-3.
cc_dsasym: online filter (0.01 < mean acc < 1.01) drops all-wrong groups before the update.
"""
import json, statistics as st, sys
from collections import defaultdict

RUNS = {"cc": "std", "cc_dsasym": "std", "cropcost": "std", "floor_l025": "floor", "floor_l050": "floor"}
CATS = ("all_correct", "mixed", "all_wrong")

def load(tag):
    groups = defaultdict(list)
    for line in open(f"grpo_v3_A_{tag}_rollouts.jsonl"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        groups[(d["step"], d["uid"])].append(d)
    return groups

def advantages(rows, mode, has_price):
    s = [r["score"] for r in rows]
    m, sd = st.mean(s), st.stdev(s)
    den = max(sd, 0.25) if mode == "floor" else sd + 1e-6
    adv = [(x - m) / den for x in s]
    if has_price and any(r["correct"] > 0 for r in rows):
        p = [r.get("price") or 0.0 for r in rows]
        mp = st.mean(p)
        adv = [a + (x - mp) for a, x in zip(adv, p)]
    return adv

for tag, mode in RUNS.items():
    groups = load(tag)
    steps = {k[0] for k in groups}
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0, 0])  # n, correct, adv, |adv|, tokens, eff_tokens
    ngroups = defaultdict(int)
    tot_abs_tok = 0.0
    abs_tok = defaultdict(float)
    for rows in groups.values():
        if len(rows) < 2:
            continue
        acc = st.mean(r["correct"] for r in rows)
        cat = "all_correct" if acc == 1 else "all_wrong" if acc == 0 else "mixed"
        ngroups[cat] += 1
        for r, a in zip(rows, advantages(rows, mode, tag.startswith("cc"))):
            k = (cat, "crop" if r["crops"] > 0 else "direct")
            x = agg[k]
            x[0] += 1; x[1] += r["correct"]; x[2] += a; x[3] += abs(a)
            x[4] += r["gen_tokens"]; x[5] += r["gen_tokens"] if abs(a) > 1e-3 else 0
            abs_tok[k] += abs(a) * r["gen_tokens"]; tot_abs_tok += abs(a) * r["gen_tokens"]
    G = sum(ngroups.values())
    print(f"\n## {tag}  (rollout steps {min(steps)}-{max(steps)}, {G} groups: " +
          ", ".join(f"{c} {ngroups[c]} ({ngroups[c]/G:.0%})" for c in CATS) + ")")
    print("| group | type | rollouts | acc | mean adv | mean |adv| | mean tokens | sup. tokens | eff. tokens | share of Σ|adv|·tok |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in CATS:
        for t in ("direct", "crop"):
            x = agg[(c, t)]
            if not x[0]:
                print(f"| {c} | {t} | 0 | | | | | | | |"); continue
            n = x[0]
            print(f"| {c} | {t} | {n} | {x[1]/n:.2f} | {x[2]/n:+.3f} | {x[3]/n:.3f} | {x[4]/n:.0f} | {x[4]} | {x[5]} | {abs_tok[(c,t)]/tot_abs_tok:.1%} |")
