"""Base-model baselines B0 (direct reasoning, no tool) and B1 (Qwen3-VL native zoom tool) on the E1 pool.

Same 552 questions (440 val + 112 V* holdout), same images, same seeds and sampling as eval_v4.py, so the records
drop into the same trade-off figure. B2 (caption protocol zero-shot) is eval_v4.py itself run on the base checkpoint.

  * direct: one user turn = image + question + a short reasoning instruction; no tools are declared.
  * native: the tool is the official Qwen3-VL `image_zoom_in_tool` schema (bbox_2d in 0-1000 relative coordinates of
    image `img_idx`, label, img_idx), declared through the model's own chat template; the crop follows Qwen-Agent's
    image_zoom_in_qwen3vl.py (32-px minimum box, smart_resize with min_pixels=256*32*32) and is returned as a
    `tool` message with the image. At most --max-crops crops; a call after the budget ends the trajectory.
  * budgets: max_model_len 32768, per-turn max_new_tokens 2048, cumulative generated tokens 8192 (docs/37 §4).
  * final answer: last <answer>X</answer>; fallbacks (\\boxed{X}, "answer is X") are scored but flagged
    format_ok=0, so task accuracy and format validity are reported separately.
Every turn is recorded; failures are written into the record. Resumable.

Usage (GPU server):
  python eval_base.py --mode direct|native --model <Qwen3-VL-4B-Instruct> --code-dir <grpo_v51> \
      --tag B0_direct --out-dir evidence/baselines_20261024 [--limit 8]
"""
from full_question import full_question
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["direct", "native"], required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--code-dir", required=True, help="only for caption_contracts.raw_image_prompt and the pool path")
ap.add_argument("--tag", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--pool", default=os.path.join(os.environ["DATA_DIR"], "grpo_caption_v51_v3"))
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--max-crops", type=int, default=8)
ap.add_argument("--seed-base", type=int, default=20261017)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, a.code_dir)
sys.path.insert(0, os.environ["EASYR1_DIR"])
sys.path.insert(0, os.path.join(os.environ["PROJECT_DIR"], "grpo_eng_repo", "eval_v4"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from verl.utils.dataset import process_image  # noqa: E402

from caption_contracts import raw_image_prompt  # noqa: E402
from traj_utils import stable_seed  # noqa: E402

MAX_LEN, MAX_GEN_TOTAL = 32768, 8192  # one budget for every runner: a turn may use all remaining generation
MIN_PX, MAX_PX = 1024, 589824
DIRECT_SUFFIX = ("\nThink step by step about the image, then give the final answer as the option letter "
                 "in the form <answer>X</answer>.")
NATIVE_SYSTEM = "Use tools to answer."
NATIVE_SUFFIX = ("\nYou may zoom in on regions of the image with the tool when it helps. When you are done, give the "
                 "final answer as the option letter in the form <answer>X</answer>.")
ZOOM_TOOL = {"type": "function", "function": {
    "name": "image_zoom_in_tool",
    "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an "
                   "optional object label",
    "parameters": {"type": "object", "properties": {
        "bbox_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                    "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) "
                                   "is the top-left corner and (x2, y2) is the bottom-right corner"},
        "label": {"type": "string", "description": "The name or label of the object in the specified bounding box"},
        "img_idx": {"type": "number", "description": "The index of the zoomed-in image (starting from 0)"}},
        "required": ["bbox_2d", "label", "img_idx"]}}}
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ANS_RE = re.compile(r"<answer>\s*\(?\s*([A-Z])\s*\)?\s*</answer>", re.I)
BOX_RE = re.compile(r"\\boxed\{\s*\(?([A-Z])\)?\s*\}")
IS_RE = re.compile(r"answer\s*(?:is|:)\s*\**\s*\(?([A-Z])\)?\b", re.I)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def extract_answer(text):
    """(letter or None, format_ok)."""
    m = ANS_RE.findall(text)
    if m:
        return m[-1].upper(), 1
    for rx in (BOX_RE, IS_RE):
        m = rx.findall(text)
        if m:
            return m[-1].upper(), 0
    return None, 0


def smart_resize(height, width, factor=32, min_pixels=56 * 56, max_pixels=12845056):
    """Qwen-Agent image_zoom_in_qwen3vl.smart_resize, verbatim logic."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar, w_bar = math.floor(height / beta / factor) * factor, math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def maybe_resize_bbox(left, top, right, bottom, w, h):
    """Qwen-Agent image_zoom_in_qwen3vl.maybe_resize_bbox, verbatim logic."""
    left, top, right, bottom = max(0, left), max(0, top), min(w, right), min(h, bottom)
    bh, bw = bottom - top, right - left
    if bh < 32 or bw < 32:
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        ratio = 32 / max(min(bh, bw), 1e-6)
        hh, hw = math.ceil(bh * ratio * 0.5), math.ceil(bw * ratio * 0.5)
        nl, nr = max(0, math.floor(cx - hw)), min(w, math.ceil(cx + hw))
        nt, nb = max(0, math.floor(cy - hh)), min(h, math.ceil(cy + hh))
        if nb - nt > 32 and nr - nl > 32:
            return [nl, nt, nr, nb]
    return [left, top, right, bottom]


def zoom(image, bbox):
    """Crop like Qwen-Agent: bbox in 0-1000 relative coords of `image`; returns (crop, abs_box)."""
    w, h = image.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"degenerate bbox {bbox}")
    box = maybe_resize_bbox(x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h, w, h)
    crop = image.crop(tuple(box))
    nw, nh = smart_resize(box[2] - box[0], box[3] - box[1], factor=32, min_pixels=256 * 32 * 32)
    return crop.resize((nw, nh), resample=Image.BICUBIC), [round(v, 1) for v in box]


df = pd.concat([pd.read_parquet(f"{a.pool}/val.parquet"), pd.read_parquet(f"{a.pool}/vstar_holdout.parquet")],
               ignore_index=True)
if a.limit:
    df = df.head(a.limit)
out_dir = Path(a.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
jsonl = out_dir / f"traj_{a.tag}.jsonl"
done = set()
if jsonl.exists():
    for line in open(jsonl):
        r = json.loads(line)
        done.add((r["uid"], r["sample_idx"]))

header = dict(tag=a.tag, mode=a.mode, model=a.model, pool=a.pool, samples=a.samples, seed_base=a.seed_base,
              n_questions=len(df), temperature=a.temperature, top_p=1.0, max_new_tokens="remaining",
              max_gen_total=MAX_GEN_TOTAL, max_model_len=MAX_LEN, min_px=MIN_PX, max_px=MAX_PX,
              max_crops=a.max_crops if a.mode == "native" else 0,
              system=NATIVE_SYSTEM if a.mode == "native" else None,
              suffix=NATIVE_SUFFIX if a.mode == "native" else DIRECT_SUFFIX,
              tool=ZOOM_TOOL if a.mode == "native" else None, vllm=vllm.__version__, torch=torch.__version__,
              gpu=torch.cuda.get_device_name(0), started=time.strftime("%F %T"),
              sha256=dict(val=sha256_file(f"{a.pool}/val.parquet"),
                          vstar_holdout=sha256_file(f"{a.pool}/vstar_holdout.parquet"),
                          model_config=sha256_file(f"{a.model}/config.json"),
                          script=sha256_file(__file__)))
(out_dir / f"header_{a.tag}.json").write_text(json.dumps(header, ensure_ascii=False, indent=1))

proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=MAX_LEN, limit_mm_per_prompt={"image": a.max_crops + 1}, seed=a.seed_base)
tools = [ZOOM_TOOL] if a.mode == "native" else None


class State:
    def __init__(self, row, s_idx):
        self.row, self.s_idx, self.uid = row, s_idx, row["uid"]
        gt = row["reward_model"]["ground_truth"] if isinstance(row["reward_model"], dict) else row["reward_model"]
        self.gt = str(gt).strip().upper()
        q = full_question(row["env_head"])
        self.raw_imgs = [Image.open(row["images"][0]).convert("RGB")]
        self.proc_imgs = [process_image(self.raw_imgs[0], MIN_PX, MAX_PX)]
        user = [{"type": "image"}, {"type": "text", "text": q + (NATIVE_SUFFIX if a.mode == "native" else DIRECT_SUFFIX)}]
        self.msgs = ([{"role": "system", "content": NATIVE_SYSTEM}] if a.mode == "native" else []) + \
            [{"role": "user", "content": user}]
        self.turns, self.done, self.final_text, self.end = [], False, None, None
        self.prompt_tokens = self.gen_tokens = self.image_tokens = 0
        self.crops, self.boxes, self.error = 0, [], None

    def record(self):
        pred, fmt = extract_answer(self.final_text or "") if self.end == "answer" else (None, 0)
        return dict(uid=self.uid, sample_idx=self.s_idx, layer=str(self.row["stratum"]),
                    source=self.row["data_source"], gt=self.gt, pred=pred, ok=int(pred is not None and pred == self.gt),
                    format_ok=fmt, n_crops=self.crops, boxes_abs=self.boxes, end=self.end, error=self.error,
                    image_wh=list(self.raw_imgs[0].size), prompt_tokens=self.prompt_tokens,
                    gen_tokens=self.gen_tokens, image_tokens=self.image_tokens, turns=self.turns,
                    final_text=self.final_text)


def finish(st, end, text=None):
    st.end, st.done = end, True
    if text is not None:
        st.final_text = text


def advance(states):
    prompts, sps, live = [], [], []
    for st in states:
        if st.done:
            continue
        try:
            text = proc.apply_chat_template(st.msgs, tools=tools, add_generation_prompt=True, tokenize=False)
            enc = proc(images=st.proc_imgs, text=[text], add_special_tokens=False, return_tensors="pt")
            ids = enc["input_ids"][0].tolist()
            budget = min(MAX_GEN_TOTAL - st.gen_tokens, MAX_LEN - len(ids))
            if budget < 64:
                finish(st, "overlong")
                continue
            st.prompt_tokens += len(ids)
            st.image_tokens += sum(1 for t in ids if t == proc.image_token_id)
            prompts.append({"prompt_token_ids": raw_image_prompt(ids, enc["image_grid_thw"].tolist(), proc),
                            "multi_modal_data": {"image": st.proc_imgs}})
            sps.append(SamplingParams(max_tokens=budget, temperature=a.temperature, top_p=1.0,
                                      seed=stable_seed(a.seed_base, st.uid, st.s_idx, len(st.turns))))
            live.append(st)
        except Exception as e:  # recorded, never swallowed
            st.error = f"prompt:{type(e).__name__}:{e}"
            finish(st, "error")
    if not live:
        return
    outs = llm.generate(prompts, sps, use_tqdm=False)
    for st, o in zip(live, outs):
        try:
            out = o.outputs[0]
            tg = proc.tokenizer.decode(list(out.token_ids), skip_special_tokens=False)
            tg = tg.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            st.gen_tokens += len(out.token_ids)
            turn = dict(round=len(st.turns), text=tg, finish=out.finish_reason, n_tokens=len(out.token_ids))
            st.turns.append(turn)
            calls = TOOL_RE.findall(tg) if a.mode == "native" else []
            if not calls:
                finish(st, "answer" if out.finish_reason != "length" else "truncated", tg)
                continue
            if st.crops >= a.max_crops:
                finish(st, "crop_budget", tg)
                continue
            st.msgs.append({"role": "assistant", "content": tg})
            turn["tools"] = []
            for raw in calls:  # the chat template allows several calls per turn: run all, in order
                try:
                    if st.crops >= a.max_crops:
                        raise ValueError("crop budget exhausted")
                    args = json.loads(raw).get("arguments", {})
                    idx = int(args.get("img_idx", 0))
                    crop, box = zoom(st.raw_imgs[idx], args["bbox_2d"])
                except Exception as e:  # invalid call: tell the model, as Qwen-Agent does
                    turn["tools"].append(dict(error=f"{type(e).__name__}:{e}"))
                    st.msgs.append({"role": "tool", "content": [{"type": "text", "text": f"Error: {e}"}]})
                    continue
                st.crops += 1
                st.boxes.append(dict(img_idx=idx, box=box, size=list(crop.size)))
                turn["tools"].append(dict(img_idx=idx, bbox_2d=args["bbox_2d"], box_abs=box))
                st.raw_imgs.append(crop)
                st.proc_imgs.append(process_image(crop, MIN_PX, MAX_PX))
                st.msgs.append({"role": "tool", "content": [{"type": "image"}]})
        except Exception as e:
            st.error = f"step:{type(e).__name__}:{e}"
            finish(st, "error")


t0, n_done, n_ok = time.time(), 0, 0
rows = [r for _, r in df.iterrows()]
max_rounds = 1 if a.mode == "direct" else 2 * a.max_crops + 2
with open(jsonl, "a") as fout:
    for c0 in range(0, len(rows), a.chunk):
        states = [State(r, s) for r in rows[c0:c0 + a.chunk] for s in range(a.samples) if (r["uid"], s) not in done]
        for _ in range(max_rounds):
            if all(s.done for s in states):
                break
            advance(states)
        for st in states:
            if not st.done:
                finish(st, "round_limit")
            rec = st.record()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            n_ok += rec["ok"]
        fout.flush()
        print(f"[{a.tag}] {min(c0 + a.chunk, len(rows))}/{len(rows)} q, {n_done} samples, acc {n_ok / max(n_done, 1):.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)
print("DONE", jsonl)
