"""Append stopped-run E1 exports using the existing scorer and public-export rules.

Usage: python export_stop40.py LOCAL_ICML_WORKSPACE
Does not modify the 21 original trajectory exports or their score table.
"""
import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import export_e1_rollouts as ex


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    workspace = Path(sys.argv[1])
    src = workspace / "obs/runs/k8_continue_20261029"
    out = ex.REPO / "logs/stopped40_20261029"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ex.REPO / "logs/baselines_v52_20261024/summary.json").read_text())
    stats = json.loads((ex.OUT / "STATS.json").read_text())
    ex.load_stems(str(src / "eval_stopped40/review_k8c_dapo_s40.jsonl"))
    receipts = []
    for c, method in zip(json.loads((src / "summary_stopped40.json").read_text()), ("DAPO", "GSPO")):
        cid = f"K-{method}-40"
        c.update(id=cid, name=f"SFT + {method}（追加至40步）", label=f"4k×8，额外训练预算，第40步")
        source = src / "eval_stopped40" / c["file"]
        raw = [json.loads(s) for s in source.read_text().splitlines()]
        rows = [ex.rollout(cid, r) for r in raw]
        assert len(rows) == len({(r['item_id'], r['sample_idx']) for r in rows}) == 2208
        assert len({r['item_id'] for r in rows}) == 552
        assert round(sum(r["correct"] for r in rows) / len(rows), 4) == c["acc"]
        for row in rows:
            for text in ex.strings(row):
                for key, pattern in ex.HARD_PATTERNS.items():
                    assert not pattern.search(text), (cid, row["item_id"], key)
                if row["item_id"] in ex.STEMS:
                    assert not ex.STEMS[row["item_id"]].search(text)
        dest = ex.OUT / f"{cid}.jsonl.gz"
        with gzip.GzipFile(filename="", fileobj=dest.open("wb"), mode="wb", mtime=0) as gz:
            gz.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
        stats[cid] = dict(name=c["name"], label=c["label"], rollouts=len(rows), acc=c["acc"],
                          steps=dict(Counter(r["steps"] for r in rows)), crops_total=sum(r["crops"] for r in rows),
                          redacted_in_assistant_text=dict(ex.REDACTED))
        ex.REDACTED.clear()
        receipts.append(dict(condition=cid, raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             public_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), checkpoint_step=40))
        summary.append(c)
    dump(ex.OUT / "STATS.json", stats)
    dump(out / "PROVENANCE.json", receipts)
    dump(out / "summary_all.json", summary)
    exact = {}
    table = ["| 条件 | 准确率 | 首裁 / 题 | 后续裁图 / 题 | 总处理 token / 题 |", "|---|---:|---:|---:|---:|"]
    for c in summary:
        cid = c['id']
        rows = [json.loads(s) for s in gzip.open(ex.OUT / f"{cid}.jsonl.gz", "rt")]
        assert len(rows) == 2208 and round(sum(r['correct'] for r in rows)/len(rows), 4) == c['acc']
        buckets = {}
        for n in sorted({r['crops'] for r in rows}):
            group = [r for r in rows if r['crops'] == n]
            buckets[str(n)] = dict(n=len(group), correct=sum(r['correct'] for r in group),
                                   accuracy=sum(r['correct'] for r in group)/len(group))
        exact[cid] = buckets
        table.append(f"| {cid} | {c['acc']:.4f} | {c['first']:.4f} | {c['later']:.4f} | {c['tokens']:.1f} |")
    dump(out / "accuracy_by_exact_crops.json", exact)
    with (out / "accuracy_by_exact_crops.csv").open('w') as f:
        writer = csv.writer(f); writer.writerow(['condition','executed_crops','n','correct','accuracy'])
        for cid, buckets in exact.items():
            for n, b in buckets.items(): writer.writerow([cid,n,b['n'],b['correct'],b['accuracy']])
    (out / "RESULTS.zh.md").write_text("# E1 统一结果（23个条件）\n\n552题×4采样，每条件2,208条。使用原评分器；40步为额外预算探索，不能当同32步预算对照。E1曾参与开发，不称独立盲测。\n\n" + "\n".join(table) + "\n\n按实际执行裁图次数分层的分子、分母和准确率见[CSV](accuracy_by_exact_crops.csv)。这是模型自行选择路径后的条件统计，不是增加裁图次数的因果收益；没有样本的档位不记0准确率。首裁为每题至少裁一次的采样比例，后续裁图为max(N−1,0)的均值。处理token累加每轮完整输入和输出，未扣prefix缓存，不等于GPU FLOPs。\n")
    print(f"Verified {len(summary)} conditions / {len(summary)*2208} rollouts")


if __name__ == "__main__":
    main()
