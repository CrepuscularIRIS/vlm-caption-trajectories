"""构建 GRPO 题库：公开基准 − SFT 已用题（按题面文本去重），渲染与 SFT 完全同分布。

运行前提: benchmarks 下载完成（mme_realworld 图 + mmmu/worldbench parquet + visreason/vstar）。
输出: ${DATA_DIR}/grpo_caption/{train,val}.parquet
列: prompt(含<image>), images[abs_path], system, env_head, env_tail, reward_model{ground_truth}
"""
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, __import__("os").environ["CAPTION_GRPO_CODE"])
import caption_env as CENV
import caption_env as CENV
proto = CENV.proto  # 跟随 CAPTION_GRPO_CODE 目录的协议

SFT = Path(__import__("os").environ["DATA_DIR"]) / "vlm-sft-v2-2"  # [scrub]
B = Path(__import__("os").environ["DATA_DIR"]) / "benchmarks"  # [scrub]
import os as _os
OUT = Path(_os.environ.get("GRPO_POOL_DIR", _os.environ["DATA_DIR"] + "/grpo_caption"))
OUT.mkdir(exist_ok=True, parents=True)


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())[:512]


# 1) SFT 用过的题面（含 mme_realworld_lite，统一按文本去重）
data = json.load(open(SFT / "data/caption_sft_v2_2_keep.json"))
used = set()
for d in data:
    h = next(c["value"] for c in d["conversations"] if c["from"] == "human")
    m = re.match(r"Question: (.*?)\n\(A\)", h, re.S)
    if m:
        used.add(norm_q(m.group(1)))
print(f"SFT used questions (normalized): {len(used)}")


def img_size(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def make_item(source: str, uid: str, image: Path, question: str, gold: str):
    try:
        W, H = img_size(image)
    except Exception as e:
        print(f"  [skip] bad image {image}: {e}")
        return None
    shown = CENV.present_size(W, H, is_original=True)
    try:
        head, tail = proto.first_user_text(question, (), (W, H), shown, CENV.MAX_CROPS, for_export=True)  # v5.1 签名
    except TypeError:
        head, tail = proto.first_user_text(question, (W, H), shown, CENV.MAX_CROPS)  # v6 签名
    return {
        "prompt": head + "\n<image>" + tail,
        "images": [str(image)],
        "system": proto.SYSTEM,
        "env_head": head,
        "env_tail": tail,
        "reward_model": {"style": "rule", "ground_truth": str(gold).strip().upper()},
        "data_source": source,
        "uid": uid,
        "extra_info": {"question": question[:200]},
    }


items = []

# 2) MME-RealWorld（23,609 题；Image 字段无扩展名，落盘为 .jpg/.png）
mme = json.load(open(B / "mme_realworld/MME_RealWorld.json"))
img_root = B / "mme_realworld"


def find_mme_image(rel: str):
    p = img_root / rel
    if p.exists():
        return p
    for ext in (".jpg", ".png", ".jpeg"):
        q = img_root / (rel + ext)
        if q.exists():
            return q
    return None


for idx, x in enumerate(mme):
    if x.get("Question Type") != "Multiple Choice":
        continue
    if x["Image"].startswith("remote_sensing"):
        continue  # RS 下载完成后补
    if norm_q(x["Text"]) in used:
        continue
    p = find_mme_image(x["Image"])
    if p is None:
        continue
    q = x["Text"].strip() + "\n" + "\n".join(x["Answer choices"])
    it = make_item("mme_realworld", f"mme_{idx}", p, q, x["Ground truth"])
    if it:
        items.append(it)
print(f"after MME(4cat): {len(items)}")

# 3) WorldBench / MMMU-Pro / VisReason / V* —— parquet/jsonl 结构在运行时核对后填充
# （占位函数： WorldBench 行结构 {question, options, answer, image(bytes)}; MMMU 同; VisReason class_N.jsonl; vstar test_questions.jsonl）
def from_parquet_embedded(parquet_path, source, q_col, opt_col, ans_col, img_cols):
    t = pq.read_table(parquet_path)
    df = t.to_pandas()
    imgdir = OUT / "images" / source
    imgdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, row in df.iterrows():
        q = str(row[q_col]).strip()
        if norm_q(q) in used:
            continue
        img_bytes = None
        for c in img_cols:
            v = row[c] if c in row else None
            if v is not None and (isinstance(v, (bytes, bytearray)) or (isinstance(v, dict) and v.get("bytes"))):
                img_bytes = v["bytes"] if isinstance(v, dict) else bytes(v)
                break
        if img_bytes is None:
            continue
        p = imgdir / f"{source}_{i}.jpg"
        if not p.exists():
            p.write_bytes(img_bytes)
        opts = row[opt_col] if opt_col in row else None
        if opts is None:
            continue
        if isinstance(opts, str):
            opts = json.loads(opts)
        if opts and not str(opts[0]).startswith("(A"):
            lines = [f"({chr(65 + j)}) {o}" for j, o in enumerate(opts)]
        else:
            lines = list(opts)
        question = q + "\n" + "\n".join(lines)
        gold = row[ans_col] if ans_col in row else None
        if gold is None:
            continue
        gold = str(gold).strip().upper()
        if len(gold) > 2:
            m = re.search(r"[A-J]", gold)
            gold = m.group(0) if m else gold
        it = make_item(source, f"{source}_{i}", p, question, gold)
        if it:
            items.append(it)
            n += 1
    print(f"{source}: +{n}")


if __name__ == "__main__":
    # WorldBench: 2000 行, 列 image(bytes)/question/option_a..d/answer
    import pandas as pd
    wb = pq.read_table(B / "worldbench/train.parquet").to_pandas()
    wdir = OUT / "images" / "worldbench"
    wdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, row in wb.iterrows():
        q = str(row["question"]).strip()
        if norm_q(q) in used:
            continue
        imgd = row["image"]
        if not isinstance(imgd, dict) or not imgd.get("bytes"):
            continue
        p = wdir / f"worldbench_{i}.png"
        if not p.exists():
            p.write_bytes(imgd["bytes"])
        question = q + "\n" + "\n".join(
            f"({chr(65 + j)}) {row[f'option_{chr(97 + j)}']}" for j in range(4))
        it = make_item("worldbench", f"worldbench_{i}", p, question, row["answer"])
        if it:
            items.append(it); n += 1
    print(f"worldbench: +{n}")

    # MMMU-Pro std(10): 仅取单图题; question 内 <image N> 占位符去除
    n = 0
    mdir = OUT / "images" / "mmmu_pro"
    mdir.mkdir(parents=True, exist_ok=True)
    for shard in sorted((B / "mmmu_pro").rglob("*.parquet")):
        df = pq.read_table(shard).to_pandas()
        for i, row in df.iterrows():
            if row["image_2"] is not None:
                continue
            imgd = row["image_1"]
            if not isinstance(imgd, dict) or not imgd.get("bytes"):
                continue
            q = str(row["question"]).strip()
            q = re.sub(r"\s*<image \d+>\s*", " ", q).strip()
            if norm_q(q) in used:
                continue
            p = mdir / f"{row['id']}.png"
            if not p.exists():
                p.write_bytes(imgd["bytes"])
            opts = row["options"]
            if isinstance(opts, str):
                import ast
                opts = ast.literal_eval(opts)
            lines = [f"({chr(65 + j)}) {o}" for j, o in enumerate(opts)]
            question = q + "\n" + "\n".join(lines)
            it = make_item("mmmu_pro", str(row["id"]), p, question, row["answer"])
            if it:
                items.append(it); n += 1
    print(f"mmmu_pro: +{n}")

    # V*: text 自带选项, label 为字母
    n = 0
    with open(B / "vstar/test_questions.jsonl") as f:
        for ln in f:
            r = json.loads(ln)
            q0 = r["text"].strip()
            qhead = q0.split("\n(A)")[0].strip()
            if norm_q(qhead) in used:
                continue
            p = B / "vstar" / r["image"]
            if not p.exists():
                continue
            it = make_item("vstar", f"vstar_{r['question_id']}", p, q0, r["label"])
            if it:
                items.append(it); n += 1
    print(f"vstar: +{n}")

    import random
    random.seed(42)
    random.shuffle(items)
    val = items[:480]
    train = items[480:]
    pd.DataFrame(train).to_parquet(OUT / "train.parquet")
    pd.DataFrame(val).to_parquet(OUT / "val.parquet")
    from collections import Counter
    print("sources:", Counter(x["data_source"] for x in items))
    print(f"total={len(items)} train={len(train)} val={len(val)}")
