"""Decision-loss MVE audit (Pro design 10-22 §8.3, layers 2 and 3b). Runs on the server, one GPU per call.

  states  --model REF   64 fixed questions: the reference policy writes turn 1 once; h = its post-caption pre-action
                        prefix; K answer + K crop-continue forks with an RNG namespace disjoint from training give
                        delta_audit(h). The first 16 questions also get the E2 recovery state: the reference first crop
                        (if it is a crop of the original image) is shifted by the E2 rule (traj_utils, as eval_v4
                        --perturb first_crop) and executed.
  score   --model M     p_M(h) = sigmoid(s_A - s_C) on every saved h, full-tag log-prob sums at T = 0.7 (HF forward,
                        logits / T as in the actor).
  recover --model M     2 continuations of every recovery state to a terminal answer under v5.2i; strict scoring.

Fork mechanics are imported from the training adapter (easyr1_caption._advance_forks / _fork_reward) so the audit
uses the same environment, protocol, budgets and reward as the training forks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=("states", "score", "recover"))
ap.add_argument("--model", required=True)
ap.add_argument("--code-dir", required=True)
ap.add_argument("--questions", required=True, help="json list of uids (order = pre-specified order)")
ap.add_argument("--pool", default=os.path.join(os.environ["DATA_DIR"], "grpo_caption_v51_v3"))  # [scrub]
ap.add_argument("--states-file", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--n-recovery", type=int, default=16)
ap.add_argument("--samples", type=int, default=2)
ap.add_argument("--seed", type=int, default=20261023)
a = ap.parse_args()

sys.path.insert(0, a.code_dir)
sys.path.insert(0, os.environ["EASYR1_DIR"])  # [scrub]
sys.path.insert(0, os.path.join(os.environ["PROJECT_DIR"], "grpo_eng_repo/eval_v4"))  # [scrub]
os.environ.setdefault("CAPTION_CROP_LAMBDA", "0.01")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import caption_env as CENV  # noqa: E402
import decision_loss as DL  # noqa: E402
import easyr1_caption as E  # noqa: E402
from caption_contracts import grid_list, raw_image_prompt, validate_images  # noqa: E402

proto = CENV.proto
MIN_PX, MAX_PX, MAX_LEN, RESP = 1024, 589824, 11264, 7168
assert proto.PROTOCOL == "v5.2i", proto.PROTOCOL


def load_questions():
    uids = json.load(open(a.questions))
    df = pd.concat([pd.read_parquet(f"{a.pool}/val.parquet"), pd.read_parquet(f"{a.pool}/vstar_holdout.parquet")],
                   ignore_index=True).set_index("uid", drop=False)
    rows = [df.loc[u] for u in uids]
    for r in rows:
        if r["system"] != proto.SYSTEM:
            raise ValueError("pool SYSTEM differs from the protocol SYSTEM")
    return rows


def gold_of(row) -> str:
    gt = row["reward_model"]["ground_truth"] if isinstance(row["reward_model"], dict) else row["reward_model"]
    return str(gt).strip().upper()


def make_llm(model):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.multimodal import MULTIMODAL_REGISTRY
    proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    llm = LLM(model=model, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=MAX_LEN, limit_mm_per_prompt={"image": 10}, seed=a.seed, enable_prefix_caching=True)
    mmp = MULTIMODAL_REGISTRY.create_processor(llm.llm_engine.model_config, tokenizer=proc.tokenizer)
    image_token = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
    base = SamplingParams(n=1, temperature=0.7, top_p=1.0, max_tokens=640, logit_bias={image_token: -100})
    shim = SimpleNamespace(inference_engine=SimpleNamespace(generate=lambda prompts, sampling_params, use_tqdm=False:
                                                            llm.generate(prompts, sampling_params, use_tqdm=False)),
                           _caption_mm_processor=mmp, config=SimpleNamespace(seed=a.seed))
    return proc, shim, base


def initial(proc, row):
    env = CENV.CropEnv(row["images"][0])
    ids, images, grid = E.encode_initial(proc, env, row["env_head"], row["env_tail"], MIN_PX, MAX_PX)
    return env, ids, images, grid


def stage_states():
    import copy
    from traj_utils import choose_perturbation, rewrite_bbox
    rows = load_questions()
    proc, shim, base = make_llm(a.model)
    tok = proc.tokenizer
    close, nl = tok.convert_tokens_to_ids(DL.THINK_CLOSE), tok.encode("\n", add_special_tokens=False)[0]
    tag_a, tag_c = tok.encode(DL.ANSWER_TAG, add_special_tokens=False), tok.encode(DL.CROP_TAG, add_special_tokens=False)
    inits = [initial(proc, r) for r in rows]
    reqs, sps = [], []
    for (env, ids, images, grid), r in zip(inits, rows):
        reqs.append(E.vllm_preflight(shim._caption_mm_processor, ids, images, grid, proc))
        sp = copy.deepcopy(base)
        sp.seed = E._fork_seed(a.seed, "audit_turn1", r["uid"])
        sps.append(sp)
    outs = shim.inference_engine.generate(reqs, sps)
    forks, states = [], []
    for (env, ids, images, grid), r, o in zip(inits, rows, outs):
        turn = list(o.outputs[0].token_ids)
        j = DL.find_boundary(turn, close, nl, tag_a[0])
        st = {"uid": r["uid"], "gold": gold_of(r), "turn1_text": tok.decode(turn, skip_special_tokens=True).strip(),
              "turn1_finish": o.outputs[0].finish_reason, "usable": j is not None}
        if j is not None:
            st.update(prefix=ids + turn[:j], j=j, tag_answer=tag_a, tag_crop=tag_c)
            for action, tag in (("answer", tag_a), ("grounding", tag_c)):
                for k in range(a.k):
                    forks.append({"owner": len(states), "action": action, "k": k, "ids": ids + turn[:j] + tag,
                                  "images": images[:1], "grid": grid[:1], "env": CENV.CropEnv(r["images"][0]),
                                  "turn_prefix": turn[:j] + tag, "turn_budget": E.TURN_MAX_TOKENS - j - len(tag),
                                  "cap": min(len(ids) + RESP, MAX_LEN), "terminal": None, "assistant": [],
                                  "gen_tokens": 0, "key": r["uid"],
                                  "seed": E._fork_seed(a.seed, "audit_fork", r["uid"], action, k)})
        states.append(st)
    for rnd in range(CENV.MAX_CROPS + 1):
        if not E._advance_forks(shim, forks, rnd, proc, base, MIN_PX, MAX_PX):
            break
    for f in forks:
        f["terminal"] = f["terminal"] or "no_answer"
        y, rr = E._fork_reward(proc, f, states[f["owner"]]["gold"])
        states[f["owner"]].setdefault("forks", []).append(
            {"action": f["action"], "k": f["k"], "terminal": f["terminal"], "y": y, "n_crops": f["env"].crops,
             "r": rr, "gen_tokens": f["gen_tokens"]})
    for st in states:
        if st["usable"]:
            ra = [f["r"] for f in st["forks"] if f["action"] == "answer"]
            rc = [f["r"] for f in st["forks"] if f["action"] == "grounding"]
            st["delta_audit"] = DL.decision_gap(ra, rc)
    # E2 recovery states for the first n questions (pre-specified by order)
    for idx, (st, r, (env, ids, images, grid)) in enumerate(zip(states, rows, inits)):
        if idx >= a.n_recovery:
            break
        rec = {"status": "no_crop"}
        try:
            parsed = proto.parse_step(st["turn1_text"])
            errs = proto.check_step(parsed, 1, 0, ["original_image"], CENV.MAX_CROPS)
        except (ValueError, KeyError, TypeError):
            parsed, errs = {}, ["parse_error"]
        if not errs and st["turn1_finish"] != "length" and parsed.get("kind") == "grounding" \
                and parsed.get("source") == "original_image":
            env2 = CENV.CropEnv(r["images"][0])

            def to_abs(b, e=env2):
                x0, y0, x1, y1 = e.boxes[0]
                req = (x0 + b[0] * (x1 - x0), y0 + b[1] * (y1 - y0), x0 + b[2] * (x1 - x0), y0 + b[3] * (y1 - y0))
                ab, _ = CENV.expand_min_crop(req, e.W, e.H)
                return tuple(round(v) for v in ab)
            seed = E._fork_seed(a.seed, "audit_perturb", r["uid"])
            p = choose_perturbation(list(parsed["bbox"]), to_abs, seed % 8)
            shown = [round(float(v), 4) for v in p["bbox"]]
            text2 = rewrite_bbox(st["turn1_text"], shown)
            try:
                same = text2 is not None and proto.parse_step(text2).get("bbox") == shown
            except (ValueError, KeyError, TypeError):
                same = False
            if same:
                rec = {"status": "ok", "turn1_rewritten": text2, "shown_bbox": shown, "asked_bbox": parsed["bbox"],
                       "iou_abs": p["iou_abs"], "perturb_ok": p["ok"]}
            else:
                rec = {"status": "rewrite_failed"}
        elif errs or st["turn1_finish"] == "length":
            rec = {"status": "turn1_invalid", "errors": errs or ["length"]}
        st["recovery"] = rec
    Path(a.out).write_text(json.dumps({"model": a.model, "seed": a.seed, "k": a.k, "protocol": proto.PROTOCOL,
                                       "states": states}, ensure_ascii=False))
    n_ok = sum(s["usable"] for s in states)
    print(f"[audit states] questions={len(states)} usable={n_ok} recovery_ok="
          f"{sum(s.get('recovery', {}).get('status') == 'ok' for s in states)}", flush=True)


def stage_score():
    from PIL import Image  # noqa: F401
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from verl.utils.dataset import process_image
    data = json.load(open(a.states_file))
    rows = {r["uid"]: r for r in load_questions()}
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True).cuda().eval()
    out = []
    for st in data["states"]:
        if not st["usable"]:
            out.append({"uid": st["uid"], "usable": False})
            continue
        env = CENV.CropEnv(rows[st["uid"]]["images"][0])
        img = process_image(env.images[0], MIN_PX, MAX_PX)
        feats = proc.image_processor(images=[img], return_tensors="pt")
        s = {}
        for name in ("tag_answer", "tag_crop"):
            ids = st["prefix"] + st[name]
            validate_images(ids, grid_list(feats["image_grid_thw"]), proc)
            x = torch.tensor([ids]).cuda()
            with torch.no_grad():
                logits = model(input_ids=x, attention_mask=torch.ones_like(x),
                               pixel_values=feats["pixel_values"].cuda(),
                               image_grid_thw=feats["image_grid_thw"].cuda()).logits[0]
            n = len(st[name])
            lp = torch.log_softmax(logits[-n - 1:-1].float() / 0.7, dim=-1)  # rows predicting the n tag tokens
            s[name] = float(sum(lp[t, ids[len(ids) - n + t]] for t in range(n)))
        z = s["tag_answer"] - s["tag_crop"]
        out.append({"uid": st["uid"], "usable": True, "s_answer": s["tag_answer"], "s_crop": s["tag_crop"], "z": z,
                    "p_answer": float(torch.sigmoid(torch.tensor(z))),
                    "mass": float(torch.exp(torch.tensor(s["tag_answer"])) + torch.exp(torch.tensor(s["tag_crop"])))})
    Path(a.out).write_text(json.dumps({"model": a.model, "scores": out}))
    print(f"[audit score] {a.model} states={sum(o['usable'] for o in out)}", flush=True)


def stage_recover():
    import copy
    from verl.utils.dataset import process_image
    data = json.load(open(a.states_file))
    rows = {r["uid"]: r for r in load_questions()}
    proc, shim, base = make_llm(a.model)
    tok = proc.tokenizer
    forks = []
    for st in data["states"]:
        rec = st.get("recovery")
        if not rec or rec["status"] != "ok":
            continue
        r = rows[st["uid"]]
        env, ids, images, grid = initial(proc, r)
        turn_ids = tok.encode(rec["turn1_rewritten"], add_special_tokens=False) + \
            [tok.convert_tokens_to_ids("<|im_end|>")]
        parsed = proto.parse_step(rec["turn1_rewritten"])
        receipt = env.step(parsed)
        image = process_image(receipt["image"], MIN_PX, MAX_PX)
        enc = proc(images=[image], text=[E.receipt_chunk(proc, receipt, turn_ids[-1])], add_special_tokens=False,
                   return_tensors="pt")
        r_ids = enc["input_ids"][0].tolist()
        r_grid = grid_list(enc["image_grid_thw"])
        for s in range(a.samples):
            forks.append({"owner": st["uid"], "action": "recover", "k": s, "ids": ids + turn_ids + r_ids,
                          "images": images + [image], "grid": grid + r_grid, "env": copy.deepcopy(env),
                          "turn_prefix": [], "turn_budget": E.TURN_MAX_TOKENS, "cap": min(len(ids) + RESP, MAX_LEN),
                          "terminal": None, "assistant": [turn_ids], "gen_tokens": 0, "key": st["uid"],
                          "seed": E._fork_seed(a.seed, "audit_recover", st["uid"], s), "gold": gold_of(r)})
    # round 0 of _advance_forks is the first turn; recovery continuations start at step 2
    for f in forks:
        f["action"] = None
    for rnd in range(1, CENV.MAX_CROPS + 1):
        if not E._advance_forks(shim, forks, rnd, proc, base, MIN_PX, MAX_PX):
            break
    out = []
    for f in forks:
        f["terminal"] = f["terminal"] or "no_answer"
        y, rr = E._fork_reward(proc, f, f["gold"])
        out.append({"uid": f["owner"], "sample": f["k"], "terminal": f["terminal"], "y": y, "n_crops": f["env"].crops,
                    "r": rr, "gen_tokens": f["gen_tokens"]})
    Path(a.out).write_text(json.dumps({"model": a.model, "recovery": out}))
    print(f"[audit recover] {a.model} continuations={len(out)} acc={sum(o['y'] for o in out) / max(1, len(out)):.3f}",
          flush=True)


t0 = time.time()
{"states": stage_states, "score": stage_score, "recover": stage_recover}[a.stage]()
print(f"[audit] stage={a.stage} secs={time.time() - t0:.0f}", flush=True)
