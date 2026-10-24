"""Overlap between an RL pool and the A-SFT data (question text, original-image path, original-image 16x16 ahash).

usage: python sft_overlap.py POOL_PARQUET OUT_JSON
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

SFT = Path(os.environ["DATA_DIR"] + "/vlm-sft-v2-2")
SFT_FILES = ["data/caption_sft_v2_2_aligned3983.json", "data/caption_sft_v2_2_keep.json"]
sys.path.insert(0, os.environ["PROJECT_DIR"] + "/pipeline/eval/baselines")
from full_question import full_question  # noqa: E402


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def stem(text: str) -> str:
    m = re.search(r"Question: (.*?)(?:\n\(A\)|\nOriginal image|\n\n|$)", text, re.S)
    return norm(m.group(1)) if m else ""


def full_mc(text: str) -> str:
    m = re.search(r"(Question: .*?)(?:\nOriginal image|\n\n|$)", text, re.S)
    return norm(m.group(1)) if m else ""


def ahash(p: str) -> str:
    im = Image.open(p).convert("L").resize((16, 16))
    px = list(im.getdata())
    m = sum(px) / len(px)
    return "".join("1" if v > m else "0" for v in px)


def sft_rows() -> pd.DataFrame:
    rows = []
    for f in SFT_FILES:
        for x in json.load(open(SFT / f)):
            h = next(c["value"] for c in x["conversations"] if c["from"] == "human")
            rows.append({"file": f, "stem": stem(h), "mc": full_mc(h), "img": str(SFT / x["images"][0])})
    d = pd.DataFrame(rows)
    d["h"] = d.img.map(ahash)
    return d


def main() -> None:
    pool_path, out = sys.argv[1], sys.argv[2]
    s = sft_rows()
    p = pd.read_parquet(pool_path)
    q = "Question: " + p.env_head.map(full_question)
    p["stem"] = q.map(stem)
    p["mc"] = q.map(full_mc)
    p["img"] = p.images.map(lambda v: v[0])
    p["h"] = p.img.map(ahash)
    hit_stem = p.stem.isin(set(s.stem) - {""})
    hit_mc = p.mc.isin(set(s.mc) - {""})
    hit_h = p.h.isin(set(s.h))
    res = {
        "pool": pool_path, "pool_n": len(p),
        "sft_rows": {f: int((s.file == f).sum()) for f in SFT_FILES},
        "sft_unique_stem": int(s.stem.nunique()), "sft_unique_img_hash": int(s.h.nunique()),
        "sft_rows_without_parsed_stem": int((s.stem == "").sum()),
        "pool_rows_without_parsed_stem": int((p.stem == "").sum()),
        "pool_hit_question_stem": int(hit_stem.sum()),
        "pool_hit_question_with_options": int(hit_mc.sum()),
        "pool_hit_original_image_ahash": int(hit_h.sum()),
        "pool_hit_any": int((hit_stem | hit_mc | hit_h).sum()),
        "pool_hit_any_by_source": p[hit_stem | hit_mc | hit_h].data_source.value_counts().to_dict(),
        "hit_uids": sorted(p[hit_stem | hit_mc | hit_h].uid.tolist()),
    }
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "hit_uids"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
