"""External baseline X2: DeepEyes-7B with its own zoom protocol, on the E1 pool (552 questions x 4 samples).

Protocol copied from the official DeepEyes eval/eval_vstar.py (github.com/Visual-Agent/DeepEyes):
  * system prompt = instruction_prompt_system (image_zoom_in_tool declared in <tools>); user turn = image +
    "Question: {question}\nOptions: \nA. ...\n" + USER_PROMPT_V2; our question text is split into stem and options;
    one added line "Answer with the option's letter from the given choices." (same line as X1) so that the
    shared scorer can read an option letter;
  * a turn containing <tool_call>{"name": ..., "arguments": {"bbox_2d": [l, t, r, b]}}</tool_call> crops the
    ORIGINAL image at those absolute pixel coordinates, resizes the crop with qwen-vl-utils smart_resize
    (factor 28, min 4*28*28, max 16384*28*28, bicubic) and returns
    user: "<tool_response>" + image + USER_PROMPT_V2 + "</tool_response>";
  * at most 11 model calls (official loop: try_count > 10 stops); the official loop ends at </answer>.
  * a tool call that fails to parse or crop is recorded as an error and ends the trajectory (official: exception ->
    status 'error').
Budget and scoring are ours, the same for every condition: max_model_len 32768, cumulative generation 8192;
temperature .7 x 4 samples like every E1 condition (official: greedy). summarize.py scores the final text.

usage: python eval_deepeyes.py --model <DeepEyes-7B> --tag X2_deepeyes --out-dir <dir> [--limit 8]
"""
import argparse
import ast
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from full_question import full_question  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--pool", default=os.path.join(os.environ["DATA_DIR"], "grpo_caption_v51_v3"))
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--seed-base", type=int, default=20261017)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402
from PIL import Image  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MAX_LEN, MAX_GEN_TOTAL = 32768, 8192
MAX_ROUNDS, MAX_IMAGES = 11, 12
FACTOR, MIN_PIXELS, MAX_PIXELS = 28, 4 * 28 * 28, 16384 * 28 * 28
SYSTEM = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""
USER_PROMPT_V2 = ("\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  "
                  "<think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> ")
LETTER_LINE = "Answer with the option's letter from the given choices."
IMG = "<|vision_start|><|image_pad|><|vision_end|>"
OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """qwen-vl-utils smart_resize as copied in DeepEyes eval_vstar.py."""
    h_bar = max(FACTOR, round(height / FACTOR) * FACTOR)
    w_bar = max(FACTOR, round(width / FACTOR) * FACTOR)
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar, w_bar = math.floor(height / beta / FACTOR) * FACTOR, math.floor(width / beta / FACTOR) * FACTOR
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar, w_bar = math.ceil(height * beta / FACTOR) * FACTOR, math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def user_text(question: str) -> str:
    """Our 'stem\\n(A) x\\n(B) y' -> DeepEyes 'Question: stem\\nOptions: \\nA. x\\nB. y\\n'."""
    stem, opts = [], []
    for line in question.split("\n"):
        m = OPT_RE.match(line.strip())
        if m:
            opts.append(f"{m.group(1)}. {m.group(2)}")
        elif opts:
            opts[-1] += "\n" + line
        else:
            stem.append(line)
    option_str = "\n" + "".join(o + "\n" for o in opts)
    return f"Question: {chr(10).join(stem)}\nOptions: {option_str}\n{LETTER_LINE}" + USER_PROMPT_V2


def parse_call(text: str) -> list:
    raw = text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = ast.literal_eval(raw)
    box = [float(v) for v in obj["arguments"]["bbox_2d"]]
    assert len(box) == 4, f"bad bbox_2d {box}"
    return box


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

header = dict(tag=a.tag, mode="deepeyes_native", model=a.model, pool=a.pool, samples=a.samples, seed_base=a.seed_base,
              n_questions=len(df), temperature=a.temperature, top_p=1.0, max_gen_total=MAX_GEN_TOTAL,
              max_model_len=MAX_LEN, max_rounds=MAX_ROUNDS, max_images=MAX_IMAGES, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
              system=SYSTEM, user_suffix=USER_PROMPT_V2, letter_line=LETTER_LINE,
              vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
              started=time.strftime("%F %T"),
              sha256=dict(val=sha256_file(f"{a.pool}/val.parquet"),
                          vstar_holdout=sha256_file(f"{a.pool}/vstar_holdout.parquet"),
                          model_config=sha256_file(f"{a.model}/config.json"), script=sha256_file(__file__)))
(out_dir / f"header_{a.tag}.json").write_text(json.dumps(header, ensure_ascii=False, indent=1))

llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=MAX_LEN, limit_mm_per_prompt={"image": MAX_IMAGES}, seed=a.seed_base)
tok = llm.get_tokenizer()
IMAGE_PAD = tok.convert_tokens_to_ids("<|image_pad|>")


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class State:
    def __init__(self, row, s_idx: int):
        self.row, self.s_idx, self.uid = row, s_idx, row["uid"]
        gt = row["reward_model"]["ground_truth"] if isinstance(row["reward_model"], dict) else row["reward_model"]
        self.gt = str(gt).strip().upper()
        self.original = Image.open(row["images"][0]).convert("RGB")  # every crop is cut from the original image
        self.inputs = [self.original]                                    # vLLM applies the model's own resize
        self.prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{IMG}"
                       f"{user_text(full_question(row['env_head']))}<|im_end|>\n<|im_start|>assistant\n")
        self.turns, self.done, self.final_text, self.end, self.error = [], False, None, None, None
        self.prompt_tokens = self.gen_tokens = self.image_tokens = 0
        self.crops, self.boxes = 0, []

    def record(self) -> dict:
        return dict(uid=self.uid, sample_idx=self.s_idx, layer=str(self.row["stratum"]),
                    source=self.row["data_source"], gt=self.gt, n_crops=self.crops, boxes_abs=self.boxes,
                    end=self.end, error=self.error, image_wh=list(self.original.size),
                    prompt_tokens=self.prompt_tokens, gen_tokens=self.gen_tokens, image_tokens=self.image_tokens,
                    turns=self.turns, final_text=self.final_text)


def finish(st: State, end: str, text: str | None = None) -> None:
    st.end, st.done = end, True
    if text is not None:
        st.final_text = text


def advance(states: list) -> None:
    prompts, sps, live = [], [], []
    for st in states:
        if st.done:
            continue
        if len(st.turns) >= MAX_ROUNDS:
            finish(st, "round_limit")
            continue
        n_prompt = len(tok(st.prompt, add_special_tokens=False)["input_ids"]) + sum(
            math.prod(smart_resize(im.height, im.width)) // (FACTOR * FACTOR) - 1 for im in st.inputs)  # expanded images
        budget = min(MAX_GEN_TOTAL - st.gen_tokens, MAX_LEN - n_prompt)
        if budget < 64:
            finish(st, "overlong")
            continue
        prompts.append({"prompt": st.prompt, "multi_modal_data": {"image": list(st.inputs)}})
        sps.append(SamplingParams(max_tokens=budget, temperature=a.temperature, top_p=1.0, stop=["<|im_end|>"],
                                  seed=stable_seed(a.seed_base, st.uid, st.s_idx, len(st.turns))))
        live.append(st)
    if not live:
        return
    outs = llm.generate(prompts, sps, use_tqdm=False)
    for st, o in zip(live, outs):
        try:
            out = o.outputs[0]
            ids = list(o.prompt_token_ids)
            st.prompt_tokens += len(ids)
            st.image_tokens += sum(1 for t in ids if t == IMAGE_PAD)
            st.gen_tokens += len(out.token_ids)
            text = out.text
            turn = dict(round=len(st.turns), text=text, finish=out.finish_reason, n_tokens=len(out.token_ids))
            st.turns.append(turn)
            if out.finish_reason == "length":
                finish(st, "truncated", text)
                continue
            if "<tool_call>" not in text or "</answer>" in text:  # official loop ends at </answer>
                finish(st, "answer", text)
                continue
            if len(st.inputs) >= MAX_IMAGES:
                finish(st, "image_limit", text)
                continue
            try:  # official: crop the original at absolute pixels; any exception ends the sample with status error
                left, top, right, bottom = parse_call(text)
                crop = st.original.crop((left, top, right, bottom))
                new_w, new_h = smart_resize((right - left), (bottom - top))  # argument order as in the official code
                crop = crop.resize((new_w, new_h), resample=Image.BICUBIC)
            except Exception as e:
                turn["tool"] = dict(error=f"{type(e).__name__}:{e}")
                st.error = f"tool:{type(e).__name__}:{e}"
                finish(st, "error", text)
                continue
            st.crops += 1
            st.boxes.append(dict(box=[left, top, right, bottom], size=list(crop.size)))
            turn["tool"] = dict(box_abs=[left, top, right, bottom])
            st.inputs.append(crop)
            st.prompt += (f"{text}<|im_end|>\n<|im_start|>user\n<tool_response>{IMG}{USER_PROMPT_V2}</tool_response>"
                          f"<|im_end|>\n<|im_start|>assistant\n")
        except Exception as e:
            st.error = f"step:{type(e).__name__}:{e}"
            finish(st, "error")


t0, n_done = time.time(), 0
rows = [r for _, r in df.iterrows()]
with open(jsonl, "a") as fout:
    for c0 in range(0, len(rows), a.chunk):
        states = [State(r, s) for r in rows[c0:c0 + a.chunk] for s in range(a.samples) if (r["uid"], s) not in done]
        for _ in range(MAX_ROUNDS + 1):
            if all(s.done for s in states):
                break
            advance(states)
        for st in states:
            if not st.done:
                finish(st, "round_limit")
            fout.write(json.dumps(st.record(), ensure_ascii=False) + "\n")
            n_done += 1
        fout.flush()
        print(f"[{a.tag}] {min(c0 + a.chunk, len(rows))}/{len(rows)} q, {n_done} samples, {time.time() - t0:.0f}s",
              flush=True)
print("DONE", jsonl)
