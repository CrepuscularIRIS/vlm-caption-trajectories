"""[2026-10-24 eval_v52] eval_v4 + one budget for every runner (32k context, 8192 total generation, a turn may use all
remaining generation) + pool default independent of the code-dir name. Use with --code-dir pipeline/grpo/v51fix.

Trajectory exporter for the v3 models (replaces eval_v3.py for behaviour analysis).

Runs the same 552 questions (440 stratified val + 112 V* holdout) x N samples as eval_v3.py, but
  * saves every turn of every trajectory (raw text, parsed fields, boxes, token counts);
  * uses a fixed seed per (uid, sample, round), so a rerun reproduces the same samples;
  * advances all questions of a chunk in lock-step rounds (one batched vLLM call per round);
  * never swallows an exception: failures are written into the record;
  * optionally perturbs the FIRST crop (--perturb first_crop): the model asks for box b on the original image, the
    environment crops a clearly different box b' (absolute IoU < 0.1 whenever the image allows it) and b' replaces b
    inside that turn's <grounding> action in the history. Nothing else of the model's text is touched, so the
    history reads like a self-made localisation error (the same shape as the v4c induced misses) and the receipt's
    region agrees with the action; the only cue is what the new view shows. A and B get the same b' for the same
    (uid, sample) whenever their first box is the same;
  * records, per turn, proto.check_step errors, and per sample both the eval_v3 score (ok) and the training-rollout
    score (ok_train_rule: any protocol violation or truncated turn = 0, exactly one <answer> over all turns).
Resumable: records already in the output jsonl are skipped.

Usage (on the GPU server, one model at a time):
  python eval_v4.py --model <path> --code-dir ${PROJECT_DIR}/grpo/v51fix --tag A_sft \
      --out-dir evidence/v3_behavior [--perturb none|first_crop] [--limit 20]
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--code-dir", required=True)
ap.add_argument("--tag", required=True, help="A_sft | A_rl | B_sft | B_rl")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--pool", default=None)
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--chunk", type=int, default=32, help="questions advanced together")
ap.add_argument("--perturb", choices=["none", "first_crop"], default="none")
ap.add_argument("--seed-base", type=int, default=20261017)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--min-px", type=int, default=1024,
                help="1024 = SFT (LLaMA-Factory image_min_pixels default) and RL (data.min_pixels); eval_v3 used 262144")
a = ap.parse_args()
sys.path.insert(0, a.code_dir)
sys.path.insert(0, os.environ["EASYR1_DIR"])
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from verl.utils.dataset import process_image  # noqa: E402

import caption_env as CENV  # noqa: E402  (protocol follows --code-dir)
from caption_contracts import raw_image_prompt  # noqa: E402
from reward_caption import _norm_answer  # noqa: E402
from traj_utils import choose_perturbation, rewrite_bbox, stable_seed  # noqa: E402

proto = importlib.import_module(CENV.proto.__name__)
MAX_LEN, MAX_GEN_TOTAL = 32768, 8192  # same budget as eval_base: a turn may use all remaining generation
MIN_PX, MAX_PX = a.min_px, 589824
KEEP_PARSED = ("kind", "step", "source", "bbox", "behavior", "verdict", "answer", "errors", "updates")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


pool = a.pool or os.path.join(os.environ["DATA_DIR"], "grpo_caption_v51_v3")  # E1 pool for every v5.1 code dir (v51, v52, ...)
df = pd.concat([pd.read_parquet(f"{pool}/val.parquet"), pd.read_parquet(f"{pool}/vstar_holdout.parquet")],
               ignore_index=True)
if a.limit:
    df = df.head(a.limit)

out_dir = Path(a.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
jsonl = out_dir / f"traj_{a.tag}{'_perturb' if a.perturb != 'none' else ''}.jsonl"
done = set()
if jsonl.exists():
    for line in open(jsonl):
        r = json.loads(line)
        if "uid" in r:
            done.add((r["uid"], r["sample_idx"]))

header = dict(
    tag=a.tag, model=a.model, code_dir=a.code_dir, pool=pool, perturb=a.perturb, samples=a.samples,
    seed_base=a.seed_base, n_questions=len(df), temperature=0.7, top_p=1.0, max_new_tokens="remaining",
    max_gen_total=MAX_GEN_TOTAL, max_model_len=MAX_LEN, min_px=MIN_PX, max_px=MAX_PX, max_crops=CENV.MAX_CROPS,
    proto_module=CENV.proto.__name__, vllm=vllm.__version__, torch=torch.__version__,
    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), started=time.strftime("%F %T"),
    sha256=dict(
        val=sha256_file(f"{pool}/val.parquet"), vstar_holdout=sha256_file(f"{pool}/vstar_holdout.parquet"),
        caption_env=sha256_file(f"{a.code_dir}/caption_env.py"), script=sha256_file(__file__),
        model_config=sha256_file(f"{a.model}/config.json")),
)
(out_dir / f"header_{a.tag}{'_perturb' if a.perturb != 'none' else ''}.json").write_text(
    json.dumps(header, ensure_ascii=False, indent=1))

proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=MAX_LEN, limit_mm_per_prompt={"image": 10}, seed=a.seed_base)


class State:
    """One (question, sample) trajectory in flight."""

    def __init__(self, row, s_idx):
        self.row, self.s_idx = row, s_idx
        self.uid = row["uid"]
        gt = row["reward_model"]["ground_truth"] if isinstance(row["reward_model"], dict) else row["reward_model"]
        self.gt = str(gt).strip().upper()
        self.env = CENV.CropEnv(row["images"][0])
        self.msgs = [{"role": "system", "content": row["system"]},
                     {"role": "user", "content": [{"type": "text", "text": row["env_head"] + "\n"},
                                                  {"type": "image"}, {"type": "text", "text": row["env_tail"]}]}]
        self.proc_imgs = [process_image(self.env.images[0], MIN_PX, MAX_PX)]
        self.turns, self.done, self.final_text = [], False, None
        self.overlong = self.error = None
        self.perturb = None
        self.prompt_tokens = self.gen_tokens = 0

    def abs_box(self, bbox, source):
        """Absolute crop the env would cut for `bbox` relative to `source` (mirrors CropEnv.step)."""
        k = CENV.resolve_source(source, len(self.env.boxes))
        x0, y0, x1, y1 = self.env.boxes[k]
        req = (x0 + bbox[0] * (x1 - x0), y0 + bbox[1] * (y1 - y0), x0 + bbox[2] * (x1 - x0), y0 + bbox[3] * (y1 - y0))
        ab, _ = CENV.expand_min_crop(req, self.env.W, self.env.H)
        return tuple(round(v) for v in ab)

    def record(self):
        n_crops = self.env.crops
        pred = None if (self.overlong or self.error) else _norm_answer(self.final_text or "")
        # training rule (easyr1_caption + assistant_reward_batch): any protocol violation or truncated turn ends the
        # rollout with reward 0; otherwise exactly one <answer> over all generated text must equal the gold letter.
        clean = not (self.overlong or self.error) and bool(self.turns) and all(
            not t.get("check_errors") and t["finish"] != "length" for t in self.turns) \
            and self.turns[-1]["parsed"].get("kind") == "answer"
        pred_train = _norm_answer("\n".join(t["text"] for t in self.turns)) if clean else None
        return dict(
            uid=self.uid, sample_idx=self.s_idx, layer=str(self.row["stratum"]), source=self.row["data_source"],
            gt=self.gt, pred=pred, ok=int(pred is not None and pred == self.gt), pred_train=pred_train,
            ok_train_rule=int(pred_train is not None and pred_train == self.gt), n_crops=n_crops,
            boxes_abs=[list(b) for b in self.env.boxes], image_wh=[self.env.W, self.env.H],
            overlong=bool(self.overlong), error=self.error, perturb=self.perturb,
            prompt_tokens=self.prompt_tokens, gen_tokens=self.gen_tokens, turns=self.turns,
            final_text=self.final_text)


def advance(states):
    """Run one lock-step round for all unfinished states."""
    prompts, sps, live = [], [], []
    for st in states:
        if st.done:
            continue
        try:
            text = proc.apply_chat_template(st.msgs, add_generation_prompt=True, tokenize=False)
            enc = proc(st.proc_imgs, [text], add_special_tokens=False, return_tensors="pt")
            ids = enc["input_ids"][0].tolist()
            budget = min(MAX_GEN_TOTAL - st.gen_tokens, MAX_LEN - len(ids))
            if budget < 64:
                st.overlong, st.done = True, True
                continue
            raw = raw_image_prompt(ids, enc["image_grid_thw"].tolist(), proc)
            st.prompt_tokens += len(ids)
            prompts.append({"prompt_token_ids": raw, "multi_modal_data": {"image": st.proc_imgs}})
            sps.append(SamplingParams(max_tokens=budget, temperature=0.7, top_p=1.0,
                                      seed=stable_seed(a.seed_base, st.uid, st.s_idx, len(st.turns))))
            live.append(st)
        except Exception as e:  # recorded, never swallowed
            st.error, st.done = f"prompt:{type(e).__name__}:{e}", True
    if not live:
        return
    outs = llm.generate(prompts, sps, use_tqdm=False)
    for st, o in zip(live, outs):
        try:
            out = o.outputs[0]
            tg = proc.tokenizer.decode(list(out.token_ids), skip_special_tokens=True).strip()
            st.gen_tokens += len(out.token_ids)
            try:
                parsed = proto.parse_step(tg)
            except Exception as e:
                parsed = {"errors": [f"parse:{type(e).__name__}"]}
            try:
                check = proto.check_step(parsed, expected_step=len(st.turns) + 1, crops_used=st.env.crops,
                                         available=list(st.env.available), budget=CENV.MAX_CROPS)
            except (ValueError, KeyError, TypeError) as e:
                check = [f"check:{type(e).__name__}"]
            turn = dict(round=len(st.turns), text=tg, finish=out.finish_reason, n_tokens=len(out.token_ids),
                        parsed={k: jsonable(parsed.get(k)) for k in KEEP_PARSED if k in parsed},
                        check_errors=check)
            is_ground = parsed.get("kind") == "grounding" and st.env.remaining > 0
            if not is_ground:
                st.final_text, st.done = tg, True
                st.turns.append(turn)
                continue
            step_in, hist_text = dict(parsed), tg
            if a.perturb == "first_crop" and st.env.crops == 0 and parsed.get("source") == "original_image" \
                    and parsed.get("bbox") is not None:
                seed = stable_seed(a.seed_base, st.uid, st.s_idx, 10_000)
                p = choose_perturbation(list(parsed["bbox"]), lambda b: st.abs_box(b, "original_image"), seed % 8)
                shown = [round(float(v), 4) for v in p["bbox"]]
                rewritten = rewrite_bbox(tg, shown)
                # the shifted box is written into the model's own action, as in the v4c induced misses, so the
                # receipt's region agrees with the action and the only cue left is what the new view shows
                if rewritten is not None and proto.parse_step(rewritten).get("bbox") == shown:
                    step_in["bbox"], hist_text = shown, rewritten
                    turn["text_in_history"] = rewritten
                    st.perturb = dict(asked_bbox=list(parsed["bbox"]), shown_bbox=shown, iou_abs=p["iou_abs"],
                                      ok=p["ok"], dir=list(p["dir"]))
                else:
                    st.perturb = dict(asked_bbox=list(parsed["bbox"]), skipped="bbox_rewrite_failed")
            r = st.env.step(step_in)
            if r is None:  # illegal source / budget: treat the text as the answer, as eval_v3 did
                st.final_text, st.done = tg, True
                st.turns.append(turn)
                continue
            turn["box_abs"] = list(st.env.boxes[-1])
            st.turns.append(turn)
            st.msgs.append({"role": "assistant", "content": hist_text})
            st.msgs.append({"role": "user", "content": [{"type": "text", "text": r["head"] + "\n"},
                                                        {"type": "image"}, {"type": "text", "text": r["tail"]}]})
            st.proc_imgs.append(process_image(st.env.images[-1], MIN_PX, MAX_PX))
        except Exception as e:
            st.error, st.done = f"step:{type(e).__name__}:{e}", True


t0, n_done, n_ok = time.time(), 0, 0
rows = [r for _, r in df.iterrows()]
with open(jsonl, "a") as fout:
    for c0 in range(0, len(rows), a.chunk):
        states = [State(r, s) for r in rows[c0:c0 + a.chunk] for s in range(a.samples)
                  if (r["uid"], s) not in done]
        for _ in range(CENV.MAX_CROPS + 1):
            if all(s.done for s in states):
                break
            advance(states)
        for st in states:
            st.done = True
            rec = st.record()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            n_ok += rec["ok"]
        fout.flush()
        print(f"[{a.tag}] {c0 + len(rows[c0:c0 + a.chunk])}/{len(rows)} questions, {n_done} new samples, "
              f"running acc {n_ok / max(n_done, 1):.3f}, {time.time() - t0:.0f}s", flush=True)
print("DONE", jsonl)
