"""Pre-specified 64 audit questions (order matters: the first 16 are the recovery questions).

Pool = E1 (val 440 + V* holdout 112). Excluded: every uid named in the public review packages (reviewed_uids.txt) and
every question whose image file also occurs in train.parquet (same-image overlap). Seeded shuffle, first 64.
E1 has been used repeatedly for development, so results on this set are a development diagnostic, not a confirmation.
"""
import json
import random
import sys

import pandas as pd

pool, out = sys.argv[1], sys.argv[2]
train = pd.read_parquet(f"{pool}/train.parquet")
e1 = pd.concat([pd.read_parquet(f"{pool}/val.parquet"), pd.read_parquet(f"{pool}/vstar_holdout.parquet")],
               ignore_index=True)
reviewed = set(open("reviewed_uids.txt").read().split())
train_images = {str(x[0]) for x in train["images"]}
keep = [r for _, r in e1.iterrows() if r["uid"] not in reviewed and str(r["images"][0]) not in train_images]
excl_rev = sum(r in reviewed for r in e1["uid"])
excl_img = sum(str(x[0]) in train_images for x in e1["images"])
rng = random.Random(20261023)
order = sorted(r["uid"] for r in keep)
rng.shuffle(order)
chosen = order[:64]
json.dump(chosen, open(out, "w"))
meta = {"pool_e1": len(e1), "excluded_reviewed": excl_rev, "excluded_train_image": excl_img, "eligible": len(keep),
        "chosen": len(chosen), "seed": 20261023,
        "strata": pd.Series([e1.set_index("uid").loc[u, "stratum"] for u in chosen]).value_counts().to_dict(),
        "sources": pd.Series([e1.set_index("uid").loc[u, "data_source"] for u in chosen]).value_counts().to_dict()}
json.dump(meta, open(out.replace(".json", "_meta.json"), "w"), indent=1)
print(json.dumps(meta))
