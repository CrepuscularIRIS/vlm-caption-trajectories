"""Trade-off figure from summarize.py output: accuracy vs (tokens / question, first-crop rate, later crops / question).

usage: python plot_tradeoff.py summary.json out.png
summary.json = list of summarize.py records, each with an added "id", "label" and display "name"
(the name is shown in a shared legend; markers distinguish conditions).
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p")
PANELS = (("tokens", "处理 token / 题"), ("first", "首裁率"), ("later", "后续裁图 / 题"))


def main() -> None:
    rows = json.load(open(sys.argv[1]))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    cmap = plt.get_cmap("tab20")
    for ax, (key, name) in zip(axes, PANELS):
        for i, r in enumerate(rows):
            ax.scatter(r[key], r["acc"], s=55, color=cmap(i % 20), marker=MARKERS[i % len(MARKERS)],
                       label=r.get("name", r["id"]), edgecolors="black", linewidths=0.4)
        ax.margins(x=0.15, y=0.08)
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("E1 准确率（552 题 × 4 次采样）")
    fig.suptitle("E1 trade-off（同一评分脚本，32k 上下文 / 8,192 累计生成）", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
