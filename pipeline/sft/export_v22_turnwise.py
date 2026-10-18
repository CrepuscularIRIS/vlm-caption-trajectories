"""V2.2 aligned 库 → 按轮展开（语义同 export_lf_turnwise.py；V2.2 无失误轮，全部监督）。"""
import json, os, re  # [scrub] os

SRC = os.environ["DATA_DIR"] + "/vlm-sft-v2-2/data/caption_sft_v2_2_aligned3983.json"
DST = os.environ["DATA_DIR"] + "/vlm-sft-v2-2/data/caption_sft_v2_2_keep_turnwise.json"

rows = json.load(open(SRC))
out = []
for x in rows:
    conv = x["conversations"]
    m = re.search(r"/obs/([^/]+)/obs0", x["images"][0])
    tid = m.group(1) if m else str(len(out))
    for k, t in enumerate(conv):
        if t["from"] != "gpt":
            continue
        prefix = conv[: k + 1]
        n_img = sum(c["value"].count("<image>") for c in prefix)
        out.append({"id": f"{tid}#t{k}", "task_id": tid, "target_turn": k,
                    "conversations": prefix, "images": x["images"][:n_img]})
json.dump(out, open(DST, "w"), ensure_ascii=False)
print(f"rows={len(out)} (期望 7047) distinct_tasks={len(set(r['task_id'] for r in out))}")
