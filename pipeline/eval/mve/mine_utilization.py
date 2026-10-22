"""Zero-GPU mining of 'existing-evidence utilization' failure candidates by a text rule (precedes a model/human check).
For every trajectory with a parsed final answer that is WRONG, compare the option TEXTS with the policy's own captions
(all <think> blocks):
  gold_stated_answered_wrong   the gold option's text appears in a caption, the final letter is not gold
                               (candidate 'read right, used wrong': option mapping / binding / goal drift)
  wrong_text_followed          the chosen (wrong) option's text appears in a caption and the gold text does not
                               (candidate reading error or misleading evidence followed)
  neither                      no option text matched
Option texts are usable only when distinctive: >= 3 characters after normalisation, not a substring of another option
and no other option a substring of it, and not a generic 'not feature / cannot determine' option. Trajectories whose
gold or chosen option is not distinctive are counted as 'not_checkable'. A text match is NOT a verified case: it is a
candidate list for the image check. Counts describe the scanned sets only.
Sets: SFT-A crop-branch and direct-branch trajectories on 2,000 RL-pool questions (fixed SFT-A), and E1 trajectories at
step 160 of r1 (RLOO lambda 0.01), r0 (RLOO lambda 0) and the GRPO orig rerun.
Writes mined_summary.json, candidates.jsonl (one line per candidate trajectory, with the rule), review_sample.json
(fixed sample: 20 SFT-A + 20 r1 gold_stated_answered_wrong candidates, seed 20261022, distinct uids) and
questions.json (uid -> question text with options, gold) for the reviewers.
Usage: python mine_utilization.py <out_dir>"""
import collections
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
POOL = os.environ["DATA_DIR"] + "/grpo_caption_v51_v3"  # [scrub]
E = "evidence"
SETS = {
    "sft_crop": [f"{E}/twobranch_20261019/traj_A_sft_crop.jsonl", f"{E}/twobranch_pool_20261020/traj_A_sft_crop.jsonl"],
    "sft_direct": [f"{E}/twobranch_20261019/traj_A_sft_direct.jsonl", f"{E}/twobranch_pool_20261020/traj_A_sft_direct.jsonl"],
    "r1_160": [f"{E}/rloo_20261021/traj_r1_160.jsonl"],
    "r0_160": [f"{E}/rloo_20261021/traj_r0_160.jsonl"],
    "orig_160": [f"{E}/dvgrpo_20261020/traj_orig160.jsonl"],
}
Q = {}
for split in ("train", "val", "vstar_holdout"):
    df = pd.read_parquet(f"{POOL}/{split}.parquet")
    for _, r in df.iterrows():
        ex = r["extra_info"]
        if isinstance(ex, str):
            import ast
            ex = ast.literal_eval(ex)
        g = r["reward_model"]
        g = g.get("ground_truth") if isinstance(g, dict) else g
        Q[r["uid"]] = dict(question=ex["question"], gold=str(g).strip().upper(), source=r["data_source"], split=split)
OPT = re.compile(r"^\(([A-Z])\)\s*(.+?)\s*$", re.M)
GENERIC = re.compile(r"not feature|does not|cannot|can't|none of|not sure|unable|no such|not (?:shown|visible|present)", re.I)


def norm(s):
    return re.sub(r"[^0-9a-zÀ-ɏ一-鿿.%$]+", " ", s.lower()).strip()


def options(q):
    opts = {L: norm(t) for L, t in OPT.findall(q)}
    raw = {L: t for L, t in OPT.findall(q)}
    ok = {}
    for L, t in opts.items():
        others = [o for M, o in opts.items() if M != L]
        ok[L] = (len(t.replace(" ", "")) >= 3 and not GENERIC.search(raw[L])
                 and not any(t in o or o in t for o in others if o))
    return opts, ok


def present(text, needle):
    return f" {needle} " in f" {text} "


summ, cands = {}, []
for name, paths in SETS.items():
    c = collections.Counter()
    by_src = collections.defaultdict(collections.Counter)
    for p in paths:
        for line in open(p):
            t = json.loads(line)
            q = Q.get(t["uid"])
            if q is None or t.get("error") or t.get("overlong"):
                continue
            c["n"] += 1
            pred = (t.get("pred") or "").strip().upper()
            if not pred:
                c["no_answer"] += 1
                continue
            if pred == q["gold"]:
                c["right"] += 1
                continue
            c["wrong"] += 1
            opts, ok = options(q["question"])
            if not ok.get(q["gold"]) or not ok.get(pred):
                c["not_checkable"] += 1
                by_src[q["source"]]["not_checkable"] += 1
                continue
            caps = norm(" ".join(re.findall(r"<think>(.*?)</think>", "\n".join(x["text"] for x in t["turns"]), re.S)))
            g_in, p_in = present(caps, opts[q["gold"]]), present(caps, opts[pred])
            rule = "gold_stated_answered_wrong" if g_in else ("wrong_text_followed" if p_in else "neither")
            c[rule] += 1
            by_src[q["source"]][rule] += 1
            if rule != "neither":
                cands.append(dict(set=name, rule=rule, uid=t["uid"], sample_idx=t["sample_idx"], source=q["source"],
                                  gold=q["gold"], pred=pred, n_crops=int(t["n_crops"]), file=p))
    checkable = c["wrong"] - c["not_checkable"]
    summ[name] = dict(counts=dict(c), share_of_checkable_wrong={k: round(c[k] / max(1, checkable), 3) for k in
                                                                 ("gold_stated_answered_wrong", "wrong_text_followed", "neither")},
                      by_source={s: dict(v) for s, v in by_src.items()})
with open(out / "candidates.jsonl", "w") as f:
    for x in cands:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
rng = random.Random(20261022)
sample = []
for name, k in (("sft_crop", 10), ("sft_direct", 10), ("r1_160", 20)):
    pool = [x for x in cands if x["set"] == name and x["rule"] == "gold_stated_answered_wrong"]
    rng.shuffle(pool)
    seen = set()
    for x in pool:
        if x["uid"] in seen:
            continue
        sample.append(x)
        seen.add(x["uid"])
        if len(seen) >= k:
            break
json.dump(sample, open(out / "review_sample.json", "w"), indent=1, ensure_ascii=False)
json.dump(Q, open(out / "questions.json", "w"), ensure_ascii=False)
json.dump(dict(rule_doc=__doc__.split("Usage")[0], sets=summ, n_candidates=len(cands), n_review_sample=len(sample)),
          open(out / "mined_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1)[:3000])
