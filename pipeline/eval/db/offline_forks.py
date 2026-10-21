"""Offline fork pass and gate (idea card step 3), no training. Input: free rollouts of an r1 checkpoint on fixed RL-pool
questions (eval_twobranch.py --split train --branch free --samples 16), i.e. G = 16 per question as in training.
For every decision state s_k with k >= 2 of every rollout (the context before turn k: initial view, earlier assistant
turns, receipts of crops 1..k-1), replay the crop environment and sample 4 answer-now forks with the prefill
'<think>\\nStep k | commit\\n' at the rollout temperature (0.7), max_tokens 640; z = 1 iff the fork parses to an answer
equal to gold and is not truncated. Writes forks_<tag>.jsonl (per decision state: uid, sample, k, z list, natural action).
Then (summary_<tag>.json), with r_i = ok_train_rule * (1 - 0.01 * n_crops), A_i = RLOO over the question's 16 samples,
b_m(s_k) = (1 - 0.01 (k-1)) mean(first m forks), c = r_i - b_m:
  Ec2_ratio[m] = E[c^2] / E[A^2] over k >= 2 turns, item-bootstrap 95% CI (2,000, seed 0), m in {1, 2, 4};
  m_now = smallest m whose E[c^2] is within one bootstrap SE of m = 4;
  lever: share of rollouts with any k >= 2 turn; own-state case shares at follow-up crop turns (before = b_4 > .5,
         after = r_i > 0): WR / RR / WW / RW; share of k >= 2 turns in uniform (all-equal-reward) groups;
  prefill check: at states where the policy answered naturally (turn k is the answer), fork accuracy vs the natural
         answer's correctness, and letter agreement;
  truncation rate of forks; gate = Ec2_ratio[m_now] CI upper bound < 1.
Usage: python offline_forks.py --model <ckpt> --traj <traj.jsonl> --code-dir <dir> --pool <dir> --out <dir> --tag <t>
"""
import argparse
import collections
import json
import os
import random
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--traj", required=True)
ap.add_argument("--code-dir", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--forks", type=int, default=4)
ap.add_argument("--lam", type=float, default=0.01)
a = ap.parse_args()
sys.path.insert(0, a.code_dir)
sys.path.insert(0, os.environ["EASYR1_DIR"])

import pandas as pd  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from verl.utils.dataset import process_image  # noqa: E402

import caption_env as CENV  # noqa: E402
import caption_v51 as proto  # noqa: E402
from caption_contracts import raw_image_prompt  # noqa: E402

MIN_PX, MAX_PX, MAX_LEN, FORK_TOKENS = 1024, 589824, 11264, 640
os.makedirs(a.out, exist_ok=True)
df = pd.read_parquet(f"{a.pool}/train.parquet")
ROWS = {r["uid"]: r for _, r in df.iterrows()}
T = [json.loads(l) for l in open(a.traj)]


def gold_of(row):
    g = row["reward_model"]
    if isinstance(g, str):
        import ast
        g = ast.literal_eval(g)
    return str(g.get("ground_truth", "") if isinstance(g, dict) else g).strip().upper()


def states_of(t):
    """(k, msgs, images, natural_kind) for every decision state k >= 2, replaying the environment."""
    row = ROWS[t["uid"]]
    env = CENV.CropEnv(row["images"][0])
    msgs = [{"role": "system", "content": row["system"]},
            {"role": "user", "content": [{"type": "text", "text": row["env_head"] + "\n"}, {"type": "image"},
                                         {"type": "text", "text": row["env_tail"]}]}]
    imgs = [process_image(env.images[0], MIN_PX, MAX_PX)]
    out = []
    for idx, turn in enumerate(t["turns"]):
        k = idx + 1
        p = turn.get("parsed") or {}
        if k >= 2:
            out.append((k, [dict(m) for m in msgs], list(imgs), p.get("kind"), p.get("answer")))
        if p.get("kind") != "grounding":
            break
        r = env.step(dict(p))
        if r is None:
            break
        msgs.append({"role": "assistant", "content": turn.get("text_in_history", turn["text"])})
        msgs.append({"role": "user", "content": [{"type": "text", "text": r["head"] + "\n"}, {"type": "image"},
                                                 {"type": "text", "text": r["tail"]}]})
        imgs.append(process_image(env.images[-1], MIN_PX, MAX_PX))
    return out


proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=MAX_LEN, limit_mm_per_prompt={"image": 10}, seed=0, enable_prefix_caching=True)
prompts, meta = [], []
for ti, t in enumerate(T):
    if t.get("error") or t.get("overlong"):
        continue
    for k, msgs, imgs, kind, ans in states_of(t):
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + f"<think>\nStep {k} | commit\n"
        enc = proc(imgs, [text], add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        if len(ids) + FORK_TOKENS > MAX_LEN:
            continue
        prompts.append({"prompt_token_ids": raw_image_prompt(ids, enc["image_grid_thw"].tolist(), proc),
                        "multi_modal_data": {"image": imgs}})
        meta.append((ti, k, kind, ans, f"<think>\nStep {k} | commit\n"))
print(f"[{a.tag}] decision states {len(prompts)}", flush=True)
rows_out = []
for c0 in range(0, len(prompts), 256):
    sps = [SamplingParams(n=a.forks, temperature=0.7, top_p=1.0, max_tokens=FORK_TOKENS, seed=1_000_003 * (c0 + j) + 7)
           for j in range(len(prompts[c0:c0 + 256]))]
    outs = llm.generate(prompts[c0:c0 + 256], sps, use_tqdm=False)
    for (ti, k, kind, ans, pre), o in zip(meta[c0:c0 + 256], outs):
        t = T[ti]
        gold = gold_of(ROWS[t["uid"]])
        z, letters, trunc = [], [], 0
        for comp in o.outputs:
            txt = pre + proc.tokenizer.decode(list(comp.token_ids), skip_special_tokens=True)
            try:
                p = proto.parse_step(txt)
            except (ValueError, KeyError, TypeError):
                p = {}
            ok = comp.finish_reason != "length" and p.get("kind") == "answer" and str(p.get("answer", "")).upper() == gold
            trunc += comp.finish_reason == "length"
            z.append(int(ok))
            letters.append(p.get("answer"))
        rows_out.append(dict(uid=t["uid"], sample_idx=t["sample_idx"], k=k, natural=kind, natural_answer=ans, z=z,
                             letters=letters, truncated=trunc))
    print(f"[{a.tag}] forks {min(c0 + 256, len(prompts))}/{len(prompts)}", flush=True)
with open(f"{a.out}/forks_{a.tag}.jsonl", "w") as f:
    for r in rows_out:
        f.write(json.dumps(r) + "\n")

# ---------------------------------------------------------------- summary
lam = a.lam
by_item = collections.defaultdict(list)
for t in T:
    by_item[t["uid"]].append(t)
rew = {}
for u, ts in by_item.items():
    for t in ts:
        rew[(u, str(t["sample_idx"]))] = float(t["ok_train_rule"]) * (1 - lam * int(t["n_crops"]))
fk = collections.defaultdict(dict)
for r in rows_out:
    fk[(r["uid"], str(r["sample_idx"]))][r["k"]] = r
items = sorted(by_item)
per_item = {m: {} for m in (1, 2, 4)}
cases = collections.Counter()
uniform_turns = all_turns = roll_with = 0
pre_acc = pre_nat = pre_agree = pre_n = 0
for u in items:
    ts = by_item[u]
    rs = [rew[(u, str(t["sample_idx"]))] for t in ts]
    S, G = sum(rs), len(rs)
    uniform = len(set(rs)) == 1
    acc = {m: [0.0, 0.0, 0] for m in (1, 2, 4)}
    for t, r in zip(ts, rs):
        A = r - (S - r) / max(1, G - 1)
        states = fk.get((u, str(t["sample_idx"])), {})
        roll_with += bool(states)
        for k, s in states.items():
            all_turns += 1
            uniform_turns += uniform
            for m in (1, 2, 4):
                b = (1 - lam * (k - 1)) * sum(s["z"][:m]) / m
                acc[m][0] += (r - b) ** 2
                acc[m][1] += A ** 2
                acc[m][2] += 1
            if s["natural"] == "grounding":
                before = (1 - lam * (k - 1)) * sum(s["z"]) / len(s["z"]) > .5
                cases[("R" if before else "W") + ("R" if r > 0 else "W")] += 1
            elif s["natural"] == "answer":
                pre_n += 1
                pre_acc += sum(s["z"]) / len(s["z"])
                pre_nat += int(str(s["natural_answer"] or "").upper() == gold_of(ROWS[u]))
                pre_agree += sum(int(x == s["natural_answer"]) for x in s["letters"]) / len(s["letters"])
    for m in (1, 2, 4):
        per_item[m][u] = acc[m]


def ratio(d, us):
    c = sum(d[u][0] for u in us)
    r = sum(d[u][1] for u in us)
    return c / r if r else float("nan")


rng = random.Random(0)
summ = dict(tag=a.tag, n_items=len(items), n_rollouts=len(T), n_decision_states=len(rows_out),
            share_rollouts_with_followup=roll_with / max(1, len(T)),
            share_followup_turns_in_uniform_groups=uniform_turns / max(1, all_turns),
            fork_truncation_rate=sum(r["truncated"] for r in rows_out) / max(1, a.forks * len(rows_out)),
            own_state_cases={c: v / max(1, sum(cases.values())) for c, v in sorted(cases.items())},
            prefill_check=dict(n=pre_n, fork_acc=pre_acc / max(1, pre_n), natural_acc=pre_nat / max(1, pre_n),
                               letter_agreement=pre_agree / max(1, pre_n)),
            Ec2_ratio={})


def ec2(d, us):
    n = sum(d[u][2] for u in us)
    return sum(d[u][0] for u in us) / n if n else float("nan")


draws = [[rng.choice(items) for _ in items] for _ in range(2000)]
for m in (1, 2, 4):
    bs = sorted(ratio(per_item[m], us) for us in draws)
    be = [ec2(per_item[m], us) for us in draws]
    mu = sum(be) / len(be)
    summ["Ec2_ratio"][m] = dict(point=ratio(per_item[m], items), lo=bs[50], hi=bs[1949],
                                Ec2_db=ec2(per_item[m], items), Ec2_db_se=(sum((x - mu) ** 2 for x in be) / (len(be) - 1)) ** 0.5)
e4 = summ["Ec2_ratio"][4]
summ["m_now"] = next(m for m in (1, 2, 4) if abs(summ["Ec2_ratio"][m]["Ec2_db"] - e4["Ec2_db"]) <= e4["Ec2_db_se"])
summ["gate_pass"] = summ["Ec2_ratio"][summ["m_now"]]["hi"] < 1
json.dump(summ, open(f"{a.out}/summary_{a.tag}.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
