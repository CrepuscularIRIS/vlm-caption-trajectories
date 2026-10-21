"""Parity smoke: the served reader (tool_credit.read_one) must reproduce the offline gate reads (frozen_reader.py,
evidence/battery_20261021/reader/reads_A_rl.jsonl) on the same view states: same argmax letter on >= 95% of states and
|confidence difference| <= 0.02 on >= 95%. Replays views exactly as frozen_reader.views().
Usage: python reader_parity.py <traj.jsonl> <reads.jsonl> <pool> <n_traj>"""
import json, os, sys
sys.path.insert(0, os.environ["CAPTION_GRPO_CODE"]); sys.path.insert(0, os.environ["EASYR1_DIR"])
import pandas as pd
from verl.utils.dataset import process_image
import caption_env as CENV
import tool_credit as TC
traj, reads, pool, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
df = pd.concat([pd.read_parquet(f"{pool}/val.parquet"), pd.read_parquet(f"{pool}/vstar_holdout.parquet")], ignore_index=True)
ROWS = {r["uid"]: r for _, r in df.iterrows()}
T = [json.loads(l) for l in open(traj)][:n]
R = [json.loads(l) for l in open(reads)][:n]
url, model = os.environ["CAPTION_READER_URL"], os.environ["CAPTION_READER_MODEL"]
same = close = tot = 0
for t, r in zip(T, R):
    if not r["reads"]:
        continue
    row = ROWS[t["uid"]]
    env = CENV.CropEnv(row["images"][0]); imgs = [process_image(env.images[0], 1024, 589824)]
    for turn in t["turns"]:
        p = turn.get("parsed") or {}
        if p.get("kind") == "grounding" and "box_abs" in turn:
            env.step(dict(p)); imgs.append(process_image(env.images[-1], 1024, 589824))
    q = TC.question_text(row["extra_info"])
    for j, (a0, c0) in enumerate(r["reads"]):
        a1, c1 = TC.read_one(q, imgs[: j + 1], url, model)
        tot += 1; same += int(a1 == a0); close += int(abs(c1 - c0) <= 0.02)
res = dict(states=tot, same_letter=same / tot, conf_within_002=close / tot)
res["pass"] = res["same_letter"] >= 0.95 and res["conf_within_002"] >= 0.95
print(json.dumps(res))
