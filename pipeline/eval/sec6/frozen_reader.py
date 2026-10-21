"""Direction C offline gate (idea card crop-efficiency-recover, method step S1, adapted to the files that exist).
A frozen Qwen3-VL-4B-Instruct reader R sees only pixels: the question with options, the views available before /
after each crop, and an assistant turn forced to '<answer>(' (the model writes '(X)'). p_R = top-20 logprobs of the first
generated token, summed over each letter's forms ('X', ' X') and renormalised over the M letters; letters outside the
top 20 get 0.

Per E1 trajectory (552 items x 4, T=0.7) with n crops, R reads view states j = 0..n (state j = initial view + crops
1..j, replayed with the crop environment from the stored actions). For follow-up crop k >= 2:
  rho_k = 1[c(V_<k) >= tau and a(V_<k) = a(V_<=k) = y_i]   (V_<k = state k-1, V_<=k = state k, y_i = final letter)
Adaptation (recorded): the card computes L / W on GRPO-A training logs; those logs keep no boxes, so groups here are
the 4 E1 samples of an item and A_i = (r_i - mean) / std over the 4 (0 when all equal). Turn tokens are the stored
n_tokens of the turn that issued crop k.
Outputs per tau in {0.7, 0.8, 0.9}:
  flag_share (of follow-up crops), by k (2, 3, 4+), flag rate on correct vs wrong rollouts;
  share_in_mixed_with_unflagged (flagged follow-ups in items that also contain unflagged follow-ups);
  pos_adv_mass_flagged (share of positive-advantage turn-token mass on follow-up turns that is flagged);
  L = share of positive-advantage follow-up token mass that is flagged AND in a mixed item that also has a correct
      rollout answering at a settled state (c(V_last) >= tau, a = gold);
  W = share of flagged follow-ups that are on wrong rollouts (A_i < 0) in mixed items;
  N_lever = (#flagged follow-ups counted in L or W) / #rollouts;
  F = mean max(0, n_crops - 1); MDE_F = 1.96 * sqrt(2) * SE_item(F) (item bootstrap, B = 1000, seed 0);
  gate = 'testable' if N_lever >= MDE_F else 'not_testable'.
Usage: python frozen_reader.py --reader <path> --pool <dir> --code-dir <dir> --out <dir> --traj name=path ...
"""
import argparse
import collections
import json
import math
import os
import random
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--reader", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--code-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--traj", nargs="+", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, a.code_dir)
sys.path.insert(0, os.environ["EASYR1_DIR"])

import pandas as pd  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from verl.utils.dataset import process_image  # noqa: E402

import caption_env as CENV  # noqa: E402
from caption_contracts import raw_image_prompt  # noqa: E402

MIN_PX, MAX_PX, MAX_LEN = 1024, 589824, 11264
TAUS = (0.7, 0.8, 0.9)
OPT = re.compile(r"^\(([A-Z])\)", re.M)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
df = pd.concat([pd.read_parquet(f"{a.pool}/val.parquet"), pd.read_parquet(f"{a.pool}/vstar_holdout.parquet")],
               ignore_index=True)
ROWS = {r["uid"]: r for _, r in df.iterrows()}
proc = AutoProcessor.from_pretrained(a.reader, trust_remote_code=True)
llm = LLM(model=a.reader, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=MAX_LEN, limit_mm_per_prompt={"image": 10}, seed=0)
SP = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)


def question_text(row) -> str:
    ex = row["extra_info"]
    if isinstance(ex, str):
        import ast
        ex = ast.literal_eval(ex)
    return ex["question"]


_LID = {}


def letter_ids(letters):
    key = tuple(letters)
    if key in _LID:
        return _LID[key]
    ids = {}
    for L in letters:
        for form in (L, " " + L):  # the letter may be generated with or without a leading space; both count
            t = proc.tokenizer.encode(form, add_special_tokens=False)
            if len(t) == 1:
                ids[t[0]] = L
        if L not in ids.values():
            raise ValueError(f"option letter {L!r} has no single-token form")
    _LID[key] = ids
    return ids


def views(t):
    """Processed view images per state j = 0..n (replayed); None if the replay disagrees with the stored boxes."""
    row = ROWS[t["uid"]]
    env = CENV.CropEnv(row["images"][0])
    imgs = [process_image(env.images[0], MIN_PX, MAX_PX)]
    for turn in t["turns"]:
        p = turn.get("parsed") or {}
        if p.get("kind") != "grounding" or "box_abs" not in turn:
            continue
        r = env.step(dict(p))
        if r is None or list(env.boxes[-1]) != list(turn["box_abs"]):
            return None
        imgs.append(process_image(env.images[-1], MIN_PX, MAX_PX))
    return imgs


def build_prompt(qtext, imgs):
    content = []
    for j in range(len(imgs)):
        content += [{"type": "text", "text": "Original image:" if j == 0 else f"Crop {j}:"}, {"type": "image"}]
    content.append({"type": "text", "text": f"\n{qtext}\nAnswer with the option letter only."})
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + "<answer>("
    enc = proc(imgs, [text], add_special_tokens=False, return_tensors="pt")
    raw = raw_image_prompt(enc["input_ids"][0].tolist(), enc["image_grid_thw"].tolist(), proc)
    return {"prompt_token_ids": raw, "multi_modal_data": {"image": imgs}}


def read_all(trajs):
    """For each trajectory: list of (a, c) per state, or None."""
    prompts, meta, res = [], [], {}
    for i, t in enumerate(trajs):
        if t["error"] or t["overlong"]:
            res[i] = None
            continue
        imgs = views(t)
        if imgs is None:
            res[i] = None
            continue
        q = question_text(ROWS[t["uid"]])
        letters = OPT.findall(q)
        for j in range(len(imgs)):
            prompts.append(build_prompt(q, imgs[: j + 1]))
            meta.append((i, j, letters))
        res[i] = [None] * len(imgs)
    for c0 in range(0, len(prompts), 512):
        outs = llm.generate(prompts[c0:c0 + 512], SP, use_tqdm=False)
        for (i, j, letters), o in zip(meta[c0:c0 + 512], outs):
            ids = letter_ids(letters)
            lp = o.outputs[0].logprobs[0]
            ps = collections.Counter()
            for k, v in lp.items():
                if k in ids:
                    ps[ids[k]] += math.exp(v.logprob)
            z = sum(ps.values())
            if z <= 0:
                res[i][j] = (None, 0.0)
                continue
            best = max(ps, key=ps.get)
            res[i][j] = (best, ps[best] / z)
        print(f"read {min(c0 + 512, len(prompts))}/{len(prompts)}", flush=True)
    return [res[i] for i in range(len(trajs))]


def adv4(ok):
    m = sum(ok) / len(ok)
    sd = (sum((x - m) ** 2 for x in ok) / (len(ok) - 1)) ** 0.5 if len(ok) > 1 else 0.0
    return [0.0 if sd == 0 else (x - m) / sd for x in ok]


def gate(trajs, reads, tau):
    items = collections.defaultdict(list)
    for i, t in enumerate(trajs):
        items[t["uid"]].append(i)
    fol = []  # (i, k, flag, A_i, tokens)
    settled_item = {}
    for u, idx in items.items():
        ok = [float(trajs[i]["ok"]) for i in idx]
        A = adv4(ok)
        mixed = 0 < sum(ok) < len(ok)
        settled_item[u] = any(
            trajs[i]["ok"] and reads[i] and reads[i][-1][0] == trajs[i]["gt"] and reads[i][-1][1] >= tau for i in idx)
        for i, Ai in zip(idx, A):
            r, t = reads[i], trajs[i]
            if not r:
                continue
            ctoks = [tt.get("n_tokens", 0) for tt in t["turns"] if (tt.get("parsed") or {}).get("kind") == "grounding"]
            for k in range(2, len(r)):
                (ab, cb), (aa, _) = r[k - 1], r[k]
                flag = int(t["pred"] is not None and ab is not None and cb >= tau and ab == aa == t["pred"])
                fol.append(dict(i=i, u=u, k=k, flag=flag, A=Ai, tok=ctoks[k - 1] if k - 1 < len(ctoks) else 0,
                                mixed=mixed, ok=t["ok"]))
    n_roll = sum(1 for r in reads if r is not None)
    nf = len(fol) or 1
    flagged = [f for f in fol if f["flag"]]
    pos = [f for f in fol if f["A"] > 0]
    pos_mass = sum(f["A"] * f["tok"] for f in pos) or 1.0
    Lset = [f for f in flagged if f["A"] > 0 and f["mixed"] and settled_item[f["u"]]]
    Wset = [f for f in flagged if f["A"] < 0 and f["mixed"]]
    by_item_unflag = {f["u"] for f in fol if not f["flag"]}
    byk = collections.defaultdict(lambda: [0, 0])
    for f in fol:
        kk = str(f["k"]) if f["k"] < 4 else "4+"
        byk[kk][0] += 1
        byk[kk][1] += f["flag"]
    corr = [f["flag"] for f in fol if f["ok"]]
    wrong = [f["flag"] for f in fol if not f["ok"]]
    return dict(
        n_followups=len(fol), flag_share=len(flagged) / nf,
        flag_share_by_k={k: dict(n=v[0], share=v[1] / v[0]) for k, v in sorted(byk.items())},
        flag_rate_correct=sum(corr) / max(1, len(corr)), flag_rate_wrong=sum(wrong) / max(1, len(wrong)),
        share_in_items_with_unflagged=sum(f["u"] in by_item_unflag for f in flagged) / max(1, len(flagged)),
        pos_adv_mass_flagged=sum(f["A"] * f["tok"] for f in pos if f["flag"]) / pos_mass,
        L=sum(f["A"] * f["tok"] for f in Lset) / pos_mass,
        W=len(Wset) / max(1, len(flagged)),
        N_lever=(len(Lset) + len(Wset)) / max(1, n_roll))  # L (A > 0) and W (A < 0) are disjoint


def mde_f(trajs):
    by = collections.defaultdict(list)
    for t in trajs:
        by[t["uid"]].append(max(0, int(t["n_crops"]) - 1))
    U = sorted(by)
    item = [sum(v) / len(v) for v in by.values()]
    rng = random.Random(0)
    bs = [sum(rng.choice(item) for _ in U) / len(U) for _ in range(1000)]
    m = sum(bs) / len(bs)
    se = (sum((x - m) ** 2 for x in bs) / (len(bs) - 1)) ** 0.5
    return dict(F=sum(item) / len(item), SE_item=se, MDE_F=1.96 * math.sqrt(2) * se)


summary = {}
for spec in a.traj:
    name, path = spec.split("=", 1)
    trajs = [json.loads(l) for l in open(path)]
    trajs = trajs[: a.limit] if a.limit else trajs
    for t in trajs:
        t["ok"] = int(t["ok"])
    reads = read_all(trajs)
    with open(out / f"reads_{name}.jsonl", "w") as f:
        for t, r in zip(trajs, reads):
            f.write(json.dumps(dict(uid=t["uid"], sample_idx=t["sample_idx"], reads=r)) + "\n")
    m = mde_f(trajs)
    s = dict(n_traj=len(trajs), n_replayed=sum(r is not None for r in reads), **m, per_tau={})
    for tau in TAUS:
        g = gate(trajs, reads, tau)
        g["gate"] = "testable" if g["N_lever"] >= m["MDE_F"] else "not_testable"
        s["per_tau"][str(tau)] = g
    summary[name] = s
    print(name, json.dumps(s)[:1500], flush=True)
json.dump(summary, open(out / "reader_summary.json", "w"), indent=1)
