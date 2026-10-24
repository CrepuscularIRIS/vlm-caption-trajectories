"""External baseline X1: Mini-o3-7B-v1 with its own zoom protocol, on the E1 pool (552 questions x 4 samples).

Protocol copied from the official Mini-o3 repo (verl/trainer/constants.py, vllm_rollout_spmd.py, README RL/eval
command), not from our caption protocol:
  * system prompt TOOL_CROP_SYSTEM_PROMPT; user turn = image + question text (with options) + QUESTION_SUFFIX;
  * the model zooms with <grounding>{"bbox_2d": [x0,y0,x1,y1], "source": "original_image"|"observation_i"}</grounding>
    in relative [0,1] coordinates (use_relative_coordinates=True); generation stops at </grounding>;
  * the crop is cut from the raw source image (use_raw_image=True, resize=1), then process_image(max 2,000,000 px,
    min 40,000 px) like every model input image; returned with TOOL_CALL_CROP_MULTI_TRUN_PROMPT; an invalid call
    returns the error text + ERROR_INFO_MULTI_TURN_PROMPT;
  * at most 12 generation rounds and 12 images (val_max_generation_round=12, val_limit_mm_per_prompt=12).
Budget and scoring are ours, the same for every condition: max_model_len 32768, cumulative generation 8192
(a turn may use the remaining budget); summarize.py scores the final text. Every turn is recorded. Resumable.

usage: python eval_minio3.py --model <Mini-o3-7B-v1> --tag X1_minio3 --out-dir <dir> [--limit 8]
"""
import argparse
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
MAX_ROUNDS, MAX_IMAGES = 12, 12
MAX_PX, MIN_PX = 2_000_000, 40_000
SYSTEM = ("You are a helpful assistant. Answer the user's question based on the image provided. Output your thinking "
          "process within the <think> and </think> tags. Whenever you find anything unclear, you can zoom in a "
          "specific region in the given image to see more clearly by outputing <grounding>{\"bbox_2d\": [x0, y0, x1, "
          "y1], \"source\": \"original_image\"}</grounding>, where (x0, y0) and (x1, y1) are the top-left and "
          "bottom-right coordinates of the region that you want to zoom in, respectively (suppose the width and "
          "height of the image are 1.0), and 'source' refers to the image that you zoom in and could be either "
          "'original_image' or 'observation_i'. Once the final answer is confirmed, put it within <answer> and "
          "</answer>.")
TOOL_PROMPT = ("After the above Action {action_turn}, here is the the zoom-in image (Observation {observation_turn}):\n"
               "<|vision_start|><|image_pad|><|vision_end|>.\nContinue your reasoning process inside <think> and "
               "</think>. If needed, you can continue to zoom in on the original image or any of the observations, by "
               "outputting <grounding> and </grounding> as before. If the final answer is confirmed, put your final "
               "answer inside <answer> and </answer>.")
ERROR_PROMPT = ("Please analyze the error information obtained from the function tool and adjust your response. "
                "Countinue your reasoning process inside <think> and </think>.")
QUESTION_SUFFIX = "\nAnswer with the option's letter from the given choices."
IMG = "<|vision_start|><|image_pad|><|vision_end|>"
GROUND_RE = re.compile(r"<grounding>(.*?)</grounding>", re.S)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def process_image(image: Image.Image) -> Image.Image:
    """Mini-o3 vllm_rollout_spmd.process_image, same logic."""
    if image.width * image.height > MAX_PX:
        f = math.sqrt(MAX_PX / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)), resample=Image.Resampling.NEAREST)
    if image.width * image.height < MIN_PX:
        f = math.sqrt(MIN_PX / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)), resample=Image.Resampling.NEAREST)
    if image.width < 28 or image.height < 28:
        f = 28 / min(image.width, image.height)
        image = image.resize((int(image.width * f + 1), int(image.height * f + 1)), resample=Image.Resampling.NEAREST)
    if image.width / image.height >= 200:
        image = image.resize((image.width, int(image.width / 190 + 1)), resample=Image.Resampling.NEAREST)
    if image.height / image.width >= 200:
        image = image.resize((int(image.height / 190 + 1), image.height), resample=Image.Resampling.NEAREST)
    return image.convert("RGB") if image.mode != "RGB" else image


def grounding(raw: str, observations: list) -> tuple[int, list[int]]:
    """Mini-o3 prepare_grounding_inputs_multi_turn with use_relative_coordinates=True; raises on an invalid call."""
    obj = json.loads(raw)
    bbox, source = obj["bbox_2d"], obj["source"]
    assert isinstance(bbox, list) and len(bbox) == 4, f"bad bbox_2d {bbox}"
    if source == "original_image":
        idx = 0
    else:
        m = re.match(r"observation_([0-9]*)", source)
        if not m:
            raise ValueError(f"The 'source' argument \"{source}\" does not match the pattern 'observation_([0-9]*)'.")
        idx = int(m.group(1))
    assert idx < len(observations), f"Error in predicting the source observation: \"{source}\"."
    w, h = observations[idx].size
    box = [int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)]
    box = [max(box[0], 0), max(box[1], 0), min(box[2], w - 1), min(box[3], h - 1)]
    assert box[0] <= box[2] - 1, f"The bounding box is not valid in width: {box}"
    assert box[1] <= box[3] - 1, f"The bounding box is not valid in height: {box}"
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw / bh >= 200 or bh / bw >= 200:
        raise ValueError("The absolute aspect ratio of the bounding box exceeds 200, which is too large.")
    return idx, box


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

header = dict(tag=a.tag, mode="minio3_native", model=a.model, pool=a.pool, samples=a.samples, seed_base=a.seed_base,
              n_questions=len(df), temperature=a.temperature, top_p=1.0, max_gen_total=MAX_GEN_TOTAL,
              max_model_len=MAX_LEN, max_rounds=MAX_ROUNDS, max_images=MAX_IMAGES, max_px=MAX_PX, min_px=MIN_PX,
              system=SYSTEM, tool_prompt=TOOL_PROMPT, error_prompt=ERROR_PROMPT, suffix=QUESTION_SUFFIX,
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
        self.observations = [Image.open(row["images"][0]).convert("RGB")]  # raw images the crops are cut from
        self.inputs = [process_image(self.observations[0])]                 # images the model sees
        q = full_question(row["env_head"])
        self.prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{IMG}{q}{QUESTION_SUFFIX}"
                       f"<|im_end|>\n<|im_start|>assistant\n")
        self.turns, self.done, self.final_text, self.end, self.error = [], False, None, None, None
        self.prompt_tokens = self.gen_tokens = self.image_tokens = 0
        self.crops, self.boxes = 0, []

    def record(self) -> dict:
        return dict(uid=self.uid, sample_idx=self.s_idx, layer=str(self.row["stratum"]),
                    source=self.row["data_source"], gt=self.gt, n_crops=self.crops, boxes_abs=self.boxes,
                    end=self.end, error=self.error, image_wh=list(self.observations[0].size),
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
            max(1, round(im.width / 28)) * max(1, round(im.height / 28)) - 1 for im in st.inputs)  # expanded images
        budget = min(MAX_GEN_TOTAL - st.gen_tokens, MAX_LEN - n_prompt)
        if budget < 64:
            finish(st, "overlong")
            continue
        prompts.append({"prompt": st.prompt, "multi_modal_data": {"image": list(st.inputs)}})
        sps.append(SamplingParams(max_tokens=budget, temperature=a.temperature, top_p=1.0, stop=["</grounding>"],
                                  include_stop_str_in_output=True,
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
            calls = GROUND_RE.findall(text)
            if not calls or not text.rstrip().endswith("</grounding>"):
                finish(st, "answer", text)
                continue
            act = turn["round"]  # official: Action {current_iteration}, Observation {current_iteration + 1}
            if len(st.inputs) >= MAX_IMAGES:
                finish(st, "image_limit", text)
                continue
            try:
                idx, box = grounding(calls[-1].strip(), st.observations)
                crop = st.observations[idx].crop(tuple(box))
            except Exception as e:  # official loop: error text goes back to the model
                turn["tool"] = dict(error=f"{type(e).__name__}:{e}")
                st.prompt += (f"{text}<|im_end|>\n<|im_start|>user\n{type(e).__name__}: {e}\n{ERROR_PROMPT}"
                              f"<|im_end|>\n<|im_start|>assistant\n")
                continue
            st.crops += 1
            st.boxes.append(dict(source_idx=idx, box=box, size=list(crop.size)))
            turn["tool"] = dict(source_idx=idx, box_abs=box)
            st.observations.append(crop)
            st.inputs.append(process_image(crop))
            st.prompt += (f"{text}<|im_end|>\n<|im_start|>user\n"
                          f"{TOOL_PROMPT.format(action_turn=act, observation_turn=act + 1)}<|im_end|>\n"
                          f"<|im_start|>assistant\n")
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
