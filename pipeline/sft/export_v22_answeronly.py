"""Answer-only SFT export (mentor baseline): library A's 3,983 aligned tasks, same first user turn and original
image, the target is only the final <answer>; no caption, no crop turns. SYSTEM = caption_ao.SYSTEM.
Also writes GRPO pools with the answer-only SYSTEM (same questions/answers/images as grpo_caption_v51_v3).

Run from ${DATA_DIR}/vlm-sft-v2-2:  python export_v22_answeronly.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "${PROJECT_DIR}/grpo/v51_ansonly")
sys.path.insert(0, "${CAPTION_REF_DIR}")
import caption_ao  # noqa: E402

DATA = Path("data")
SRC = DATA / "caption_sft_v2_2_aligned3983.json"
OUT_NAME = "caption_sft_v2_2_answer_only"
POOL_SRC = Path("${DATA_DIR}/grpo_caption_v51_v3")
POOL_OUT = Path("${DATA_DIR}/grpo_caption_ao_v3")
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def export_sft() -> dict:
    rows, skipped = [], []
    for x in json.loads(SRC.read_text()):
        conv = x["conversations"]
        assert conv[0]["from"] == "system" and conv[1]["from"] == "human" and conv[1]["value"].count("<image>") == 1
        last = [c for c in conv if c["from"] == "gpt"][-1]["value"]
        found = ANSWER.findall(last)
        if len(found) != 1:
            skipped.append(x.get("id") or x.get("task_id"))
            continue
        rows.append({"id": x.get("id"), "task_id": x.get("task_id"),
                     "conversations": [{"from": "system", "value": caption_ao.SYSTEM}, conv[1],
                                       {"from": "gpt", "value": f"<answer>{found[0].strip()}</answer>"}],
                     "images": x["images"][:1]})
    (DATA / f"{OUT_NAME}.json").write_text(json.dumps(rows, ensure_ascii=False))
    info_path = DATA / "dataset_info.json"
    info = json.loads(info_path.read_text())
    if OUT_NAME not in info:
        shutil.copy(info_path, DATA / "dataset_info.json.bak_20261018_ao")
        info[OUT_NAME] = dict(info["caption_sft_v2_2_keep_turnwise"], file_name=f"{OUT_NAME}.json")
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return {"rows": len(rows), "skipped_no_single_answer": skipped}


def export_pools() -> dict:
    import pandas as pd
    POOL_OUT.mkdir(exist_ok=True)
    out = {}
    for f in sorted(POOL_SRC.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["system"] = caption_ao.SYSTEM
        df.to_parquet(POOL_OUT / f.name, index=False)
        out[f.name] = len(df)
    return out


if __name__ == "__main__":
    print(json.dumps({"sft": export_sft(), "pools": export_pools(), "protocol": caption_ao.PROTOCOL_VERSION}))
