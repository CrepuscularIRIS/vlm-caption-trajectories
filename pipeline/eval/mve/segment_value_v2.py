"""Segment-value MVE v2 (measurement fix after t-segment-value; zero parameter updates).
Changes vs v1, nothing else: (1) cuts are token-aligned: the turn text is tokenised once with the policy tokenizer,
each character anchor snaps LEFT to the last token boundary at or before it, and the prefix is the token ids of the
prompt plus turn_ids[:k] (no re-tokenisation of a cut string); (2) the control P1m keeps the fact sentence and the
action structure and replaces only the gold option's text inside it by "the text" (cases where the gold text is not
found verbatim in the fact sentence get no P1m; counted). Prefixes: P0, P1t, P1m, P2t.

v1 docstring:
Segment-value MVE (zero parameter updates): does the value of continuing rise when the student writes a correct
fact, and how much value is there right before the decisive error (option mapping / binding / goal drift)?
Cases: the 13 eligible cases of the utilization image review (9 core + 4 mixed) with sentence-level anchors
(segment_boundaries.json: fact and error character ranges inside the rendered turn text). Each case is continued under
the policy that generated it (SFT-A for sft_* sets, r1 step 160 for r1_160). Prefixes per case:
  P0   history up to the start of the fact turn (the assistant turn not begun)
  P1   P0 + the fact turn's text up to the end of the fact sentence
  P1c  P1 with the fact sentence replaced by a neutral sentence of the same role (control; length differs, recorded)
  P2   history + text up to the start of the decisive error (inside the error turn)
From every prefix, M continuations at the rollout temperature (0.7) follow the original protocol: the current turn is
completed, crops are executed in the replayed crop environment (shared budget 8, turn cap 640 tokens, context 11,264),
until an answer or a stop condition. Success = parsed final letter == gold (Y); also R = Y (1 - 0.01 N) with N the total
crops including those already in the prefix. Writes runs.jsonl (one row per continuation) and summary.json:
per case and prefix V = mean Y (and mean R), and over the 9 core cases (paired by case, bootstrap 2,000, seed 0):
  S1 = mean[V(P1) - V(P0)], S2 = mean[V(P1) - V(P1c)], S3 = mean V(P2); the 4 mixed cases separately.
Usage: python segment_value.py --cases <segment_boundaries.json> --policy sft|r1 --model <ckpt> --out <dir> [--m 16]"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cases", required=True)
ap.add_argument("--policy", required=True, choices=["sft", "r1"])
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--m", type=int, default=16)
ap.add_argument("--sample", required=True, help="review_sample.json (case u{n:02d} = entry n; gives the trajectory file)")
ap.add_argument("--code-dir", required=True)
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

MIN_PX, MAX_PX, MAX_LEN, TURN_MAX = 1024, 589824, 11264, 640
MASK = "the text"
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
POOL = os.environ["DATA_DIR"] + "/grpo_caption_v51_v3"
ROWS = {}
for split in ("train", "val", "vstar_holdout"):
    for _, r in pd.read_parquet(f"{POOL}/{split}.parquet").iterrows():
        ROWS[r["uid"]] = r
SAMPLE = json.load(open(a.sample))
CASES = [c for c in json.load(open(a.cases)) if (c["case"]["set"].startswith("sft") if a.policy == "sft" else c["case"]["set"] == "r1_160")]


def traj_of(c):
    src = SAMPLE[int(c["id"][1:])]
    assert src["uid"] == c["case"]["uid"] and str(src["sample_idx"]) == str(c["case"]["sample_idx"]), c["id"]
    for line in open(src["file"]):
        t = json.loads(line)
        if t["uid"] == c["case"]["uid"] and str(t["sample_idx"]) == str(c["case"]["sample_idx"]):
            return t
    raise KeyError(c["id"])


def state_at(t, row, upto_turn):
    """msgs, processed images and env after replaying turns 1..upto_turn-1 (complete turns + receipts)."""
    env = CENV.CropEnv(row["images"][0])
    msgs = [{"role": "system", "content": row["system"]},
            {"role": "user", "content": [{"type": "text", "text": row["env_head"] + "\n"}, {"type": "image"},
                                         {"type": "text", "text": row["env_tail"]}]}]
    imgs = [process_image(env.images[0], MIN_PX, MAX_PX)]
    for turn in t["turns"][: upto_turn - 1]:
        p = turn.get("parsed") or {}
        if p.get("kind") != "grounding":
            break
        r = env.step(dict(p))
        if r is None:
            raise ValueError("replay failed")
        msgs.append({"role": "assistant", "content": turn.get("text_in_history", turn["text"])})
        msgs.append({"role": "user", "content": [{"type": "text", "text": r["head"] + "\n"}, {"type": "image"},
                                                 {"type": "text", "text": r["tail"]}]})
        imgs.append(process_image(env.images[-1], MIN_PX, MAX_PX))
    return msgs, imgs, env


def gold_text(c):
    import re
    m = re.search(r"^\(" + c["case"]["gold"] + r"\)\s*(.+?)\s*$", c["question"], re.M)
    return m.group(1).strip() if m else None


def snap(ids_offsets, anchor):
    """number of tokens whose end offset <= anchor (left snap)."""
    return sum(1 for (_, e) in ids_offsets if e <= anchor)


def prefixes(c, t, row):
    """name -> (turn_no, token ids of the partial assistant turn)."""
    ft, et = c["fact_turn"], c["decisive_error_turn"]
    f0, f1 = c["fact_char_range_in_rendered_turn"]
    ftext = t["turns"][ft - 1]["text"]
    enc = proc.tokenizer(ftext, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    res = {"P0": (ft, []), "P1t": (ft, ids[:snap(offs, f1)])}
    g = gold_text(c)
    fact = ftext[f0:f1]
    pos = fact.lower().find(g.lower()) if g else -1
    if g and len(g) >= 3 and pos >= 0:
        masked = ftext[:f0] + fact[:pos] + MASK + fact[pos + len(g):]
        res["P1m"] = (ft, proc.tokenizer(masked, add_special_tokens=False)["input_ids"])
    if et and c.get("error_char_range_in_rendered_turn"):
        e0 = c["error_char_range_in_rendered_turn"][0]
        etext = t["turns"][et - 1]["text"]
        enc2 = proc.tokenizer(etext, return_offsets_mapping=True, add_special_tokens=False)
        res["P2t"] = (et, enc2["input_ids"][:snap(enc2["offset_mapping"], e0)])
    return res


proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=MAX_LEN, limit_mm_per_prompt={"image": 10}, seed=0, enable_prefix_caching=True)


def run_from(c, t, row, turn_no, partial, seeds):
    """M continuations from (history before turn_no) + partial assistant text; returns list of (Y, N, n_tokens)."""
    live = []
    for s in seeds:
        msgs, imgs, env = state_at(t, row, turn_no)
        live.append(dict(msgs=msgs, imgs=imgs, env=env, partial=list(partial), step=turn_no, done=False, y=0, toks=0, seed=s))
    for _ in range(CENV.MAX_CROPS + 2):
        todo = [x for x in live if not x["done"]]
        if not todo:
            break
        prompts, sps = [], []
        for x in todo:
            text = proc.apply_chat_template(x["msgs"], add_generation_prompt=True, tokenize=False)
            enc = proc(x["imgs"], [text], add_special_tokens=False, return_tensors="pt")
            ids = enc["input_ids"][0].tolist() + x["partial"]
            if len(ids) + TURN_MAX > MAX_LEN:
                x["done"] = True
                continue
            prompts.append({"prompt_token_ids": raw_image_prompt(ids, enc["image_grid_thw"].tolist(), proc),
                            "multi_modal_data": {"image": x["imgs"]}})
            sps.append(SamplingParams(max_tokens=TURN_MAX, temperature=0.7, top_p=1.0, seed=x["seed"] * 101 + x["step"]))
        todo = [x for x in todo if not x["done"]]
        if not prompts:
            break
        for x, o in zip(todo, llm.generate(prompts, sps, use_tqdm=False)):
            comp = o.outputs[0]
            full = proc.tokenizer.decode(x["partial"] + list(comp.token_ids), skip_special_tokens=True)
            x["toks"] += len(comp.token_ids)
            x["partial"] = []
            if comp.finish_reason == "length":
                x["done"] = True
                continue
            try:
                p = proto.parse_step(full.strip())
            except (ValueError, KeyError, TypeError):
                x["done"] = True
                continue
            if p.get("kind") == "answer":
                x["y"] = int(str(p.get("answer", "")).strip().upper() == c["case"]["gold"])
                x["done"] = True
                continue
            if p.get("kind") != "grounding" or x["env"].remaining <= 0:
                x["done"] = True
                continue
            r = x["env"].step(p)
            if r is None:
                x["done"] = True
                continue
            x["msgs"].append({"role": "assistant", "content": full.strip()})
            x["msgs"].append({"role": "user", "content": [{"type": "text", "text": r["head"] + "\n"}, {"type": "image"},
                                                         {"type": "text", "text": r["tail"]}]})
            x["imgs"].append(process_image(x["env"].images[-1], MIN_PX, MAX_PX))
            x["step"] += 1
    return [(x["y"], x["env"].crops, x["toks"]) for x in live]


rows_out = []
for ci, c in enumerate(CASES):
    t = traj_of(c)
    row = ROWS[c["case"]["uid"]]
    for name, (turn_no, partial) in prefixes(c, t, row).items():
        seeds = [2_000_003 * (ci + 1) + 10_007 * {"P0": 1, "P1t": 2, "P1m": 3, "P2t": 4}[name] + j for j in range(a.m)]
        for j, (y, n, toks) in enumerate(run_from(c, t, row, turn_no, partial, seeds)):
            rows_out.append(dict(id=c["id"], group=c["group"], policy=a.policy, prefix=name, j=j, y=y, crops=n,
                                 r=y * (1 - 0.01 * n), gen_tokens=toks, partial_tokens=len(partial)))
        print(f"[{a.policy}] {c['id']} {name} done", flush=True)
with open(out / f"runs_{a.policy}.jsonl", "w") as f:
    for r in rows_out:
        f.write(json.dumps(r) + "\n")
print(json.dumps(dict(policy=a.policy, cases=len(CASES), runs=len(rows_out))))
