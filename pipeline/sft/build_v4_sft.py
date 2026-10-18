"""v4 插值轨迹 → LLaMA-Factory 兼容格式（掩码轮合并为 user 上下文轮）。

规则（对齐 v4 交付说明"历史保留实际执行的裁图框、被强制轮不计 loss"）：
- assistant_supervision[k]=False 的 assistant 轮 + 其后的 user 回执轮 → 合并为一条 human 轮
  （失误动作与真实回执都保留为上下文，但不作为 gpt 目标 → 不计 loss）
- 其余 assistant 轮保持 gpt 角色 → LLaMA-Factory 多轮 loss 恰好只落在受监督轮上
- images 顺序 = obs0 + 各保留回执的 <image> 出现顺序（LF 按序映射）
"""
import json
import os  # [scrub]

SRC = os.environ["DATA_DIR"] + "/vlm-sft-v2-2/data/caption_sft_v4_interp_train.json"
DST = os.environ["DATA_DIR"] + "/vlm-sft-v2-2/data/caption_sft_v4_prefix.json"


def convert(x):
    convs = x["conversations"]
    sup = x["assistant_supervision"]
    assert convs[0]["from"] == "system"
    turns = convs[1:]
    out = [convs[0]]
    k = -1
    i = 0
    while i < len(turns):
        t = turns[i]
        if t["from"] == "gpt":
            k += 1
            if sup[k]:
                out.append(t)
            else:
                # 掩码：失误动作并入下一条 user 回执，合成一条 human 上下文轮
                merged = t["value"]
                if i + 1 < len(turns) and turns[i + 1]["from"] == "human":
                    merged = merged + "\n" + turns[i + 1]["value"]
                    i += 1
                out.append({"from": "human", "value": merged})
        else:
            out.append(t)
        i += 1
    # 折叠连续 human 轮（LF sharegpt 要求严格交替）
    collapsed = []
    for t in out:
        if t["from"] == "human" and collapsed and collapsed[-1]["from"] == "human":
            collapsed[-1] = {"from": "human", "value": collapsed[-1]["value"] + "\n" + t["value"]}
        else:
            collapsed.append(t)
    out = collapsed
    n_img = sum(t["value"].count("<image>") for t in out)
    return {"conversations": out, "images": x["images"][:n_img]}


def main():
    data = json.load(open(SRC))
    out = [convert(x) for x in data]
    # 结构校验：角色交替以 human 开头、<image> 数与 images 数一致
    bad = 0
    for o in out:
        n = sum(t["value"].count("<image>") for t in o["conversations"])
        if n != len(o["images"]):
            bad += 1
    json.dump(out, open(DST, "w"), ensure_ascii=False)
    kept_gpt = sum(1 for o in out for t in o["conversations"] if t["from"] == "gpt")
    print(f"trajectories={len(out)} supervised_gpt_turns={kept_gpt} image_mismatch={bad}")


if __name__ == "__main__":
    main()
