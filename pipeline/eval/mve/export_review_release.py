"""Export the two reviewed cohorts with the exact reviewed PNG views.

CPU only. Original assistant turns/actions are retained verbatim; independent
annotations remain separate. This is a publication tool, not a model reviewer.
Usage: python pipeline/eval/mve/export_review_release.py --workspace /path/to/icml
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--image-root", type=Path,
                    help="Downloaded review PNGs under receipt/ and utilization/; defaults to workspace/research/review_images_20261022")
    args = ap.parse_args()
    research = args.workspace / "research"
    trajroot = research / "trajectories_20261022"
    out = ROOT / "data/review_20261022"
    annroot = research / "annotations"
    image_root = args.image_root or research / "review_images_20261022"
    media = ROOT / "media/review_20261022"
    media.mkdir(parents=True, exist_ok=True)
    sources = {
        "evidence/twobranch_20261019": trajroot / "sft_twobranch_300",
        "evidence/twobranch_pool_20261020": trajroot / "sft_twobranch_1700",
    }
    cache = {}

    def index(path):
        if path not in cache:
            rows = [json.loads(line) for line in path.open() if line.strip()]
            cache[path] = {(r["uid"], str(r["sample_idx"])): r for r in rows}
            assert len(cache[path]) == len(rows), path
        return cache[path]

    specs = [
        ("receipt", "mve_prep_20261022_codex", "annotation.csv"),
        ("utilization", "utilization_20261022_codex", "annotation_template.csv"),
    ]
    states = read(ROOT / "logs/mve/states.json")["states"]
    sample = read(trajroot / "utilization_mining/review_sample.json")
    cases = read(trajroot / "utilization_mining/cases.json")
    anchors = read(annroot / "utilization_20261022_codex/segment_boundaries.json")
    anchor_map = {a["id"]: {k: v for k, v in a.items() if k != "question"} for a in anchors}
    summary = read(ROOT / "logs/mve/segval_summary.json")["per_case"]
    allrows, inputs = [], {}
    for cohort, directory, csvname in specs:
        ad = annroot / directory
        with (ad / csvname).open(encoding="utf-8-sig", newline="") as f:
            annotations = list(csv.DictReader(f))
        assert len(annotations) == 40
        target = out / cohort
        target.mkdir(parents=True, exist_ok=True)
        with (target / "annotations.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(annotations[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(annotations)
        hsname = "server_sha256.json" if cohort == "receipt" else "remote_image_hashes.json"
        hs = read(ad / hsname)
        image_records = hs.get("files", hs.get("images", []))
        image_records = [r for r in image_records if r["path"].endswith(".png")]
        for name in [csvname, hsname, "summary.json"]:
            inputs[f"annotations/{directory}/{name}"] = digest(ad / name)
        dump(target / "annotation_snapshot.json", read(ad / "summary.json"))
        for i, annotation in enumerate(annotations):
            cid = annotation["id"]
            meta = states[i] if cohort == "receipt" else cases[i]
            assert meta["id"] == cid
            key = (meta["uid"], str(meta["sample_idx"]))
            if cohort == "receipt":
                matches = [(p / "traj_A_sft_crop.jsonl", index(p / "traj_A_sft_crop.jsonl")[key])
                           for p in sources.values() if key in index(p / "traj_A_sft_crop.jsonl")]
                assert len(matches) == 1, (cid, len(matches))
                path, t = matches[0]
            else:
                src = Path(sample[i]["file"])
                parent = sources.get(str(src.parent), trajroot / "e1_step160")
                path = parent / src.name
                t = index(path)[key]
                assert t["pred"] == meta["pred"] and str(t["gt"]) == str(meta["gold"])
            inputs[str(path.relative_to(research))] = digest(path)
            # Explicit allowlist: never export questions, system/user prompts,
            # raw local paths or credentials. Reviewed PNGs are copied separately
            # only after matching their original review-time SHA-256 hashes.
            fields = ["uid", "sample_idx", "source", "gt", "pred", "ok", "branch",
                      "n_crops", "boxes_abs", "image_wh", "gen_tokens", "error", "overlong"]
            turnfields = ["round", "text", "text_in_history", "parsed", "box_abs",
                          "finish", "n_tokens", "check_errors"]
            record = {
                "id": cid, "cohort": cohort, "case": meta,
                "trajectory": {k: t[k] for k in fields if k in t},
                "turns": [{k: turn[k] for k in turnfields if k in turn} for turn in t["turns"]],
                "annotation": annotation,
                "reviewed_views": [r for r in image_records if r["path"].startswith(cid + "/")],
                "source_file": str(path.relative_to(research)), "source_sha256": digest(path),
                "evidence_status": "complete_model_reviewed_not_human_gold",
                "image_bytes_in_release": True,
            }
            for view in record["reviewed_views"]:
                source = image_root / cohort / view["path"]
                assert source.stat().st_size == view["bytes"], source
                assert digest(source) == view["sha256"], source
                destination = media / (view["sha256"] + ".png")
                if not destination.exists():
                    shutil.copyfile(source, destination)
                assert digest(destination) == view["sha256"]
                view["public_image"] = str(destination.relative_to(ROOT))

            def picture(filename, label):
                view = next(v for v in record["reviewed_views"] if v["path"] == f"{cid}/{filename}")
                return f"![{label}](../../../{view['public_image']})"

            if cohort == "utilization" and cid in anchor_map:
                record["segment_anchor"] = anchor_map[cid]
                record["v1_measurement"] = summary[cid]
                a = anchor_map[cid]
                # Raw turn text is the source of truth; fail instead of silently
                # accepting a rendered-text offset that points elsewhere.
                for label, tn, span in [("fact", a["fact_turn"], a["fact_char_range_in_rendered_turn"]),
                                         ("error", a["decisive_error_turn"], a["error_char_range_in_rendered_turn"])]:
                    assert t["turns"][tn - 1]["text"][span[0]:span[1]] == a[label + "_verbatim"], cid
            allrows.append(record)
            lines = [f"# {cid} · {cohort}", "", "[返回审阅目录](../README.md)", "",
                     "模型辅助审阅，非人工 gold。下方图片为本次审阅使用的学生输入视图或标明的T/C对照；与审阅时PNG哈希一致。题库完整原文未分发。", "",
                     f"数据源：`{meta['source']}`；题目 ID：`{meta['uid']}`；采样编号：`{meta['sample_idx']}`。", "",
                     "## 独立标注", ""]
            lines += [f"- **{k}**：{v}" for k, v in annotation.items() if k != "id"]
            lines += ["", "## 学生原轨迹", "", "caption 出现在本轮动作及其回执之前；本轮裁图结果只能被后续轮次使用。", ""]
            lines += ["### 初始视图（学生输入版本）", "", picture("view_0.png", f"{cid} 初始视图 original_image"), "",
                      "图片以页面宽度显示；点击图片可查看原始PNG。未额外锐化、压缩或重新裁图。", ""]
            crop_no = 0
            for n, turn in enumerate(t["turns"], 1):
                lines += [f"### 第 {n} 轮", "", "````text", turn["text"], "````", ""]
                if "box_abs" in turn:
                    lines += [f"执行框（原图像素）：`{turn['box_abs']}`。", ""]
                if (turn.get("parsed") or {}).get("kind") == "grounding":
                    crop_no += 1
                    if cohort == "utilization" or crop_no < meta["focal_crop"]:
                        lines += [f"**本次动作之后收到的回执 observation_{crop_no}**（下一轮才可使用）：", "",
                                  picture(f"view_{crop_no}.png", f"{cid} observation_{crop_no}"), ""]
                    elif crop_no == meta["focal_crop"]:
                        lines += ["**焦点回执对照**：T 是学生实际收到的图；C 从此前已看过的视图中裁出，供诊断比较，未作为这条旧轨迹的回执。", "",
                                  "| T：实际回执 | C：已有像素对照 |", "|---|---|",
                                  f"| {picture('T.png', cid + ' T 实际回执')} | {picture('C.png', cid + ' C 诊断对照')} |", ""]
                    else:
                        lines += ["此后续裁图的回执不在本批焦点审阅材料中，未补造图片；下方保留旧续写文字。", ""]
            if "v1_measurement" in record:
                lines += ["## v1 续跑结果（每个前缀 16 次）", "", "| 前缀 | 答对数 | 继续答对概率 |", "|---|---:|---:|"]
                for p in ["P0", "P1", "P1c", "P2"]:
                    v = record["v1_measurement"][p]
                    lines += [f"| {p} | {round(v['V']*v['n'])}/{v['n']} | {v['V']:.4f} |"]
                lines += ["", "前缀定义及测量局限见[导师说明](../../../docs/30_rl_credit_report_20261022.md)。"]
            lines += ["", "## 本次实际审阅的视图索引", "", "公开图片与原审阅PNG逐字节一致；哈希用于确认材料身份，不替代看图。", "",
                      "| 视图 | SHA-256 |", "|---|---|"]
            lines += [f"| {v['path']} | `{v['sha256']}` |" for v in record["reviewed_views"]]
            if cohort == "receipt":
                lines += ["", "仅焦点裁图及之前视图、T/C 经此轮看图审阅；后续轮次文字保留，不代表后续全部回执经此次视觉确认。"]
            (target / f"{cid}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert len(allrows) == 80
    with (out / "reviewed_trajectories.jsonl").open("w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dump(out / "segment_boundaries.json", list(anchor_map.values()))
    dump(out / "source_manifest.json", {"source_hashes": inputs, "n_review_records": 80,
         "n_unique_trajectories": len({(r['source_file'], r['case']['uid'], str(r['case']['sample_idx'])) for r in allrows}),
         "scope": "Public caption/action traces, annotations, anchors and exact reviewed PNG views; full benchmark questions omitted."})
    rows = ["# 80 个审阅记录：轨迹与逐题标注", "", "先读[进展报告](../../docs/30_rl_credit_report_20261022.md)，再按病例查看。", "",
            "s00–s39 检验获取新细节；u00–u39 检验利用已有证据。两批可能复用同一题/轨迹，不能当作 80 个独立样本。", "",
            "本目录公开学生完整可见 caption、裁图动作、独立标注、13 个分段锚点及审阅图片。按用户追加要求补图；题库完整原文仍不分发。", "",
            "每例按初始图→caption/动作→回执排列。利用证据批次有129张视图引用；焦点回执批次有147张（含T/C），共276处引用、274个独立PNG。图片按SHA-256去重保存，与原审阅文件逐字节一致。", "",
            "s批次只包含焦点裁图及之前的视图和T/C，不含后续未审阅回执；u批次包含完整轨迹各次回执。C是诊断对照，不是学生旧轨迹中的输入。图片来自记录所注明的数据源，项目代码许可不改变原图的来源权利。", "",
            "`annotation_snapshot.json` 保留标注完成时状态，其中 probe_executed=false 是历史快照；其后的 v1 测量见 logs/mve/segval_v1。", "",
            "| 病例 | 审阅结论/类型 | 诊断分层 |", "|---|---|---|"]
    for r in allrows:
        a = r["annotation"]
        category = next(v for k, v in a.items() if k.startswith(("category (", "failure_type (")))
        group = r.get("segment_anchor", {}).get("group", "未入选分段诊断" if r['cohort'] == 'utilization' else "配对回执门槛未过")
        rows.append(f"| [{r['id']}]({r['cohort']}/{r['id']}.md) | {category} | {group} |")
    (out / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Exported {len(allrows)} reviewed records to {out}")


if __name__ == "__main__":
    main()
