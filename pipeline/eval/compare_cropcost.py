"""Crop-cost comparison on eval_v4 trajectories: any tags against A_rl / A_sft, text output only.

Usage: python compare_cropcost.py --traj-dir evidence/v3_behavior --pool-ann evidence/v3_behavior/POOL_ANNOTATIONS.json \
           --tags A_sft A_rl A_fl025 A_fl050 [--e2-tags A_rl A_fl025 A_fl050]
Reuses analyze_v4 (features, recovery, cluster bootstrap); adds redundant-crop share and the SFT direct-correct subset.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v4 as av  # noqa: E402
from traj_utils import iou  # noqa: E402

REDUNDANT_IOU = 0.7
DIRECT_CORRECT_MIN = 3  # SFT samples (of 4) that answered directly and correctly


def redundant(r: dict) -> tuple[int, int]:
    """(crops overlapping an earlier crop with IoU > 0.7, crops) for one trajectory."""
    crops = r["boxes_abs"][1:]
    hits = sum(any(iou(c, p) > REDUNDANT_IOU for p in crops[:i]) for i, c in enumerate(crops))
    return hits, len(crops)


def e1(traj_dir: Path, tags: list[str]) -> pd.DataFrame:
    rows = []
    for tag in tags:
        for r in av.load(traj_dir / f"traj_{tag}.jsonl"):
            hits, n = redundant(r)
            rows.append(dict(tag=tag, uid=r["uid"], s=r["sample_idx"], layer=r["layer"], red_hits=hits,
                             red_n=n, red_any=int(hits > 0), **av.features(r)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True)
    ap.add_argument("--pool-ann", type=Path, required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--e2-tags", nargs="*", default=[])
    a = ap.parse_args()
    cluster = pd.Series({r["uid"]: r["image_key"] for r in json.load(open(a.pool_ann))})
    df = e1(a.traj_dir, a.tags)

    tot = df.groupby("tag").agg(n=("ok", "size"), acc=("ok", "mean"), direct=("direct", "mean"),
                                crops=("n_crops", "mean"), tokens=("gen_tokens", "mean"),
                                invalid=("invalid", "mean"), red_traj=("red_any", "mean"),
                                red_hits=("red_hits", "sum"), red_n=("red_n", "sum"))
    tot["red_crop"] = tot.red_hits / tot.red_n.clip(lower=1)
    print("## E1 overall\n" + tot.drop(columns=["red_hits", "red_n"]).round(3).to_string())
    print("\n## E1 accuracy by layer\n" + df.pivot_table(index="layer", columns="tag", values="ok").round(3).to_string())
    print("\n## E1 direct rate by layer\n" + df.pivot_table(index="layer", columns="tag", values="direct").round(3).to_string())

    if "A_sft" in a.tags:
        sft = df[df.tag == "A_sft"]
        dc = sft.assign(dc=sft.direct * sft.ok).groupby("uid")["dc"].sum()
        subset = set(dc[dc >= DIRECT_CORRECT_MIN].index)
        sub = df[df.uid.isin(subset)].groupby("tag").agg(acc=("ok", "mean"), direct=("direct", "mean"),
                                                        tokens=("gen_tokens", "mean"))
        print(f"\n## SFT direct-correct subset ({len(subset)} questions)\n" + sub.round(3).to_string())
        rest = df[~df.uid.isin(subset)].groupby("tag").agg(acc=("ok", "mean"), direct=("direct", "mean"))
        print("\n## Other questions\n" + rest.round(3).to_string())

    av.CONTRASTS = [(t, b) for t in a.tags for b in ("A_rl", "A_sft") if t not in ("A_rl", "A_sft") and b in a.tags]
    rows = av.contrasts(df, ["ok", "direct", "n_crops", "gen_tokens"], cluster)
    print("\n## E1 contrasts (95% cluster bootstrap)\n" + pd.DataFrame(rows).round(4).to_string(index=False))

    if a.e2_tags:
        av.TAGS = a.e2_tags
        e2, cov = av.e2_frame(a.traj_dir)
        print("\n## E2 coverage\n" + json.dumps(cov))
        print("\n## E2 overall\n" + e2.groupby("tag")[["ok", "drop", "recrop", "back_to_target", "false_confirm",
                                                          "extra_crops"]].mean().round(3).to_string())
        av.CONTRASTS = [(t, "A_rl") for t in a.e2_tags if t != "A_rl"]
        rows = av.contrasts(e2, ["recrop", "back_to_target", "drop"], cluster)
        print("\n## E2 contrasts vs A_rl (95% cluster bootstrap)\n" + pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
