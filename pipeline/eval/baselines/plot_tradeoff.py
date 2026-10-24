"""Trade-off figure from summarize.py output: accuracy vs (tokens / question, first-crop rate, later crops / question).

usage: python plot_tradeoff.py summary.json out.png
summary.json = list of summarize.py records, each with an added "id" (e.g. "B0") and "label".
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = (("tokens", "处理 token / 题"), ("first", "首裁率"), ("later", "后续裁图 / 题"))


def main() -> None:
    rows = json.load(open(sys.argv[1]))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (key, name) in zip(axes, PANELS):
        for r in rows:
            ax.scatter(r[key], r["acc"], s=40)
            ax.annotate(r["id"], (r[key], r["acc"]), textcoords="offset points", xytext=(5, 4), fontsize=9)
        ax.margins(x=0.15, y=0.08)
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("E1 准确率（552 题 × 4 次采样）")
    fig.suptitle("E1 trade-off（同一评分脚本，32k 上下文 / 8,192 累计生成）", fontsize=11)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=150)


if __name__ == "__main__":
    main()
