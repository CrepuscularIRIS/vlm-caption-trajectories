"""B6 data: same questions/images as a caption pool, prompt = '<image>' + question + eval_base DIRECT_SUFFIX.

usage: python build_direct_pool.py TRAIN_PARQUET VAL_PARQUET DST_DIR
"""
import sys
from pathlib import Path

import pandas as pd
from full_question import full_question

DIRECT_SUFFIX = ("\nThink step by step about the image, then give the final answer as the option letter "
                 "in the form <answer>X</answer>.")
train, val, dst = sys.argv[1], sys.argv[2], Path(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
for name, path in (("train", train), ("val", val)):
    d = pd.read_parquet(path)
    q = d["env_head"].map(full_question)
    out = pd.DataFrame({"prompt": "<image>" + q + DIRECT_SUFFIX, "images": d["images"],
                        "answer": d["reward_model"], "uid": d["uid"], "data_source": d["data_source"]})
    out.to_parquet(dst / f"{name}.parquet")
    print(name, len(out))
