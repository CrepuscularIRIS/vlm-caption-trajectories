"""E16 overlap audit (CPU only): does any E1 item (val 440 + V* holdout 112) also appear in library A's SFT data
(4,040 trajectories) or in the RL training pool (2,000 questions)?
Three keys per item: task uid (from the dataset uid / library-A observation path), normalised question text (lower case,
whitespace collapsed, options included), and a 64-bit perceptual hash of the original image (DCT pHash on a 32x32
grayscale resize, 8x8 low-frequency block vs its median); image matches at Hamming distance <= 4 and <= 8 are reported.
Writes overlap.json: counts per E1 benchmark and key, and the list of overlapping E1 uids (any key, Hamming <= 4).
Usage: python overlap_audit.py <libA_json> <libA_root> <pool_dir> <out.json>"""
import json
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

lib_json, lib_root, pool, dst = sys.argv[1:5]


def norm_q(s: str) -> str:
    s = s.split("\nOriginal image")[0]
    s = re.sub(r"^Question:\s*", "", s.strip())
    return re.sub(r"\s+", " ", s).strip().lower()


def phash(path: str) -> int:
    img = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    c = dctn(np.asarray(img, dtype=np.float64), norm="ortho")[:8, :8].flatten()
    med = np.median(c[1:])
    return int("".join("1" if v > med else "0" for v in c), 2)


def ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


UID = re.compile(r"/obs/([^/]+)/obs0\.")
lib = json.load(open(lib_json))
A = []
for x in lib:
    imgs = x["images"] if isinstance(x["images"], list) else eval(x["images"])  # noqa: S307 (stored as a list repr)
    m = UID.search(imgs[0])
    q = next(c["value"] for c in x["conversations"] if c["from"] == "human")
    A.append(dict(uid=m.group(1) if m else None, q=norm_q(q), h=phash(f"{lib_root}/{imgs[0]}")))
print("library A", len(A), flush=True)


def pool_rows(split):
    df = pd.read_parquet(f"{pool}/{split}.parquet")
    out = []
    for _, r in df.iterrows():
        ex = r["extra_info"] if isinstance(r["extra_info"], dict) else eval(r["extra_info"])  # noqa: S307
        out.append(dict(uid=str(r["uid"]), src=r["data_source"], q=norm_q(ex["question"]), h=phash(r["images"][0])))
    return out


T = pool_rows("train")
E = pool_rows("val") + pool_rows("vstar_holdout")
print("pool", len(T), "E1", len(E), flush=True)


def canon(uid):  # library-A ids look like mme_realworld_19526; pool uids like mme_19526 / test_History_239
    if uid is None:
        return None
    return uid.replace("mme_realworld_", "mme_").replace("mmmu_pro_", "").replace("worldbench_", "worldbench_")


def matches(e, ref):
    u = any(canon(r["uid"]) == canon(e["uid"]) for r in ref if r["uid"])
    q = any(r["q"] == e["q"] for r in ref)
    d = min((ham(r["h"], e["h"]) for r in ref), default=64)
    return u, q, d


res = defaultdict(lambda: defaultdict(int))
hits = []
for e in E:
    for name, ref in (("libA", A), ("rl_pool", T)):
        u, q, d = matches(e, ref)
        res[e["src"]][f"{name}_uid"] += u
        res[e["src"]][f"{name}_question"] += q
        res[e["src"]][f"{name}_img_h4"] += d <= 4
        res[e["src"]][f"{name}_img_h8"] += d <= 8
        if u or q or d <= 4:
            hits.append(dict(uid=e["uid"], src=e["src"], ref=name, uid_match=u, question_match=q, min_hamming=d))
    res[e["src"]]["n"] += 1
# pool vs library A (the RL pool is drawn from the same task pool as library A: expected overlap, reported for scope)
pa = sum(any(canon(r["uid"]) == canon(t["uid"]) for r in A if r["uid"]) for t in T)
out = dict(per_benchmark={k: dict(v) for k, v in res.items()}, n_e1=len(E), n_overlapping_e1=len({h["uid"] for h in hits}),
           overlapping_e1=hits, rl_pool_items_in_libA_by_uid=pa, n_pool=len(T), n_libA=len(A))
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "overlapping_e1"}, indent=1))
