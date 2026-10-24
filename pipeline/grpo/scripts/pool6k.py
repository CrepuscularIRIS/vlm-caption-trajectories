"""6k RL pool: pool8k.py rule (same filters, old 2000 first, rest seed42 order, ahash dedup) at n=6000,
plus: drop any row matching A-SFT data by question stem, question+options, or original-image 16x16 ahash.

usage: python pool6k.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from sft_overlap import ahash, full_mc, full_question, sft_rows, stem  # noqa: E402

N = 6016  # 47 x rollout_batch 128: drop_last=True then skips no question
B = os.environ["DATA_DIR"]
OUT = f"{B}/grpo_caption_v51_6k"
big = pd.read_parquet(f"{B}/grpo_caption_v51/train.parquet")
old = pd.read_parquet(f"{B}/grpo_caption_v51_v3/train.parquet")
e1 = pd.concat([pd.read_parquet(f"{B}/grpo_caption_v51_v3/val.parquet"),
                pd.read_parquet(f"{B}/grpo_caption_v51_v3/vstar_holdout.parquet")])
v51val = pd.read_parquet(f"{B}/grpo_caption_v51/val.parquet")
e1_uid = set(e1.uid)
e1_img = set(e1.images.map(lambda v: v[0]))
e1_hash = {ahash(p) for p in e1_img}
big["img"] = big.images.map(lambda v: v[0])
keep = big[(big.data_source != "vstar") & ~big.uid.isin(e1_uid) & ~big.img.isin(e1_img)
           & ~big.uid.isin(set(v51val.uid))].copy()
keep["h"] = keep.img.map(ahash)
keep = keep[~keep.h.isin(e1_hash)]
n_after_e1 = len(keep)

s = sft_rows()
q = "Question: " + keep.env_head.map(full_question)
keep["stem"], keep["mc"] = q.map(stem), q.map(full_mc)
sft_hit = keep.stem.isin(set(s.stem) - {""}) | keep.mc.isin(set(s.mc) - {""}) | keep.h.isin(set(s.h))
n_sft_hit = int(sft_hit.sum())
keep = keep[~sft_hit]

old_in = keep[keep.uid.isin(old.uid)]
rest = keep[~keep.uid.isin(old.uid)].sample(frac=1, random_state=42).drop_duplicates("h")
sel = pd.concat([old_in, rest.head(N - len(old_in))])
assert len(sel) == N, len(sel)
sel8k = pd.read_parquet(f"{B}/grpo_caption_v51_8k/train.parquet")
info = dict(
    n=len(sel), sources=sel.data_source.value_counts().to_dict(),
    candidates_after_e1_filters=n_after_e1, candidates_removed_sft_overlap=n_sft_hit,
    old2000_kept=len(old_in), unique_img_hash=int(sel.h.nunique()),
    in_8k_pool=int(sel.uid.isin(sel8k.uid).sum()),
    rule="pool8k rule at n=6016 (47x128) + drop A-SFT overlap (question stem / question+options / original-image ahash)",
)
sel = sel.drop(columns=["img", "h", "stem", "mc"]).sample(frac=1, random_state=42).reset_index(drop=True)
os.makedirs(OUT, exist_ok=True)
sel.to_parquet(f"{OUT}/train.parquet")
json.dump(info, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print(json.dumps(info, indent=1))
