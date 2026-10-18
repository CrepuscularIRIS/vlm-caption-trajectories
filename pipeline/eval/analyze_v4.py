"""E1 / E2 behaviour report from eval_v4 trajectories.

Usage: python analyze_v4.py --traj-dir evidence/v3_behavior --pool-ann POOL_ANNOTATIONS.json --out report_v4
Writes <out>.md (tables) and <out>.json (all numbers). Differences carry 95% cluster-bootstrap intervals, clusters =
image_key (all questions on one image are resampled together); an interval containing 0 is reported as "no difference".
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from traj_metrics import features, recovery

TAGS = ["A_sft", "A_rl", "B_sft", "B_rl"]
E1_METRICS = ["ok", "ok_train", "n_crops", "direct", "multi", "reloc", "shift", "widen", "zoom_in",
              "back_to_original", "invalid", "overlong", "budget_out", "gen_tokens", "said_miss"]
E2_METRICS = ["ok", "drop", "recrop", "back_to_target", "false_confirm", "extra_crops", "said_miss"]
CONTRASTS = [("A_sft", "B_sft"), ("A_rl", "B_rl"), ("A_rl", "A_sft"), ("B_rl", "B_sft")]
N_BOOT, SEED = 10_000, 20261017


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)] if path.exists() else []


def bootstrap_diff(x: pd.Series, y: pd.Series, cluster: pd.Series, rng: np.random.Generator) -> dict:
    """Mean of per-question (x - y) over questions present in both, with a cluster bootstrap 95% interval."""
    d = (x - y).dropna()
    if d.empty:
        return dict(diff=None, lo=None, hi=None, n=0)
    g = pd.DataFrame({"d": d, "c": cluster.reindex(d.index).fillna(pd.Series(d.index, index=d.index))})
    agg = g.groupby("c")["d"].agg(["sum", "count"])
    s, n = agg["sum"].to_numpy(), agg["count"].to_numpy()
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(diff=float(d.mean()), lo=float(lo), hi=float(hi), n=int(len(d)))


def e1_frame(traj_dir: Path) -> pd.DataFrame:
    rows = []
    for tag in TAGS:
        for r in load(traj_dir / f"traj_{tag}.jsonl"):
            rows.append(dict(tag=tag, uid=r["uid"], s=r["sample_idx"], layer=r["layer"], source=r["source"],
                             **features(r)))
    return pd.DataFrame(rows)


def e2_frame(traj_dir: Path) -> tuple[pd.DataFrame, dict]:
    rows, coverage = [], {}
    for tag in TAGS:
        base = {(r["uid"], r["sample_idx"]): r for r in load(traj_dir / f"traj_{tag}.jsonl")}
        pert = load(traj_dir / f"traj_{tag}_perturb.jsonl")
        n_used = 0
        for r in pert:
            f = recovery(r, base.get((r["uid"], r["sample_idx"])))
            if f is None:
                continue
            n_used += 1
            f["drop"] = None if f["ok_base"] is None else f["ok_base"] - f["ok"]
            rows.append(dict(tag=tag, uid=r["uid"], s=r["sample_idx"], layer=r["layer"], **f))
        coverage[tag] = dict(samples=len(pert), perturbed=n_used)
    return pd.DataFrame(rows), coverage


def table(df: pd.DataFrame, metrics: list[str], by: str) -> pd.DataFrame:
    out = df.groupby(["tag", by])[metrics].mean().round(3)
    total = df.groupby("tag")[metrics].mean().round(3)
    total[by] = "ALL"
    return pd.concat([out.reset_index(), total.reset_index()]).set_index(["tag", by]).sort_index()


def contrasts(df: pd.DataFrame, metrics: list[str], cluster: pd.Series) -> list[dict]:
    rng = np.random.default_rng(SEED)
    per_q = df.groupby(["tag", "uid"])[metrics].mean()
    out = []
    for x, y in CONTRASTS:
        if x not in per_q.index.get_level_values(0) or y not in per_q.index.get_level_values(0):
            continue
        for m in metrics:
            r = bootstrap_diff(per_q.loc[x][m], per_q.loc[y][m], cluster, rng)
            zero = r["lo"] is not None and r["lo"] <= 0 <= r["hi"]
            out.append(dict(contrast=f"{x} - {y}", metric=m, **r, verdict="no difference" if zero else "differs"))
    return out


def difficulty_link(e1: pd.DataFrame) -> dict:
    """Spearman between per-question crops and a model-independent difficulty (1 - accuracy over all 16 samples)."""
    diff = 1 - e1.groupby("uid")["ok"].mean()
    out = {}
    for tag, g in e1.groupby("tag"):
        crops = g.groupby("uid")["n_crops"].mean()
        both = pd.concat([crops, diff], axis=1, keys=["c", "d"]).dropna()
        tiers = pd.qcut(both["d"].rank(method="first"), 3, labels=["easy", "middle", "hard"])
        out[tag] = dict(spearman=round(float(both["c"].corr(both["d"], method="spearman")), 3),
                        crops_by_tier=both.groupby(tiers, observed=True)["c"].mean().round(2).to_dict())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--pool-ann", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cluster = pd.Series({r["uid"]: r["image_key"] for r in json.load(open(a.pool_ann))})
    e1 = e1_frame(Path(a.traj_dir))
    e2, coverage = e2_frame(Path(a.traj_dir))
    rep = dict(n_samples=e1.groupby("tag").size().to_dict(), e2_coverage=coverage,
               difficulty=difficulty_link(e1) if not e1.empty else {},
               e1_contrasts=contrasts(e1, E1_METRICS, cluster) if not e1.empty else [],
               e2_contrasts=contrasts(e2, [m for m in E2_METRICS if m in e2], cluster) if not e2.empty else [])
    md = ["# eval_v4 behaviour report", "", "## E1 by layer", "", table(e1, E1_METRICS, "layer").to_markdown(), "",
          "## E1 by crop bucket (accuracy)", "",
          e1.assign(bucket=e1.n_crops.clip(upper=2)).pivot_table("ok", "tag", "bucket").round(3).to_markdown(), "",
          "## Steps vs difficulty", "", "```", json.dumps(rep["difficulty"], indent=1), "```", "",
          "## E1 contrasts (95% cluster bootstrap)", "", pd.DataFrame(rep["e1_contrasts"]).round(3).to_markdown()]
    if not e2.empty:
        md += ["", "## E2 perturbed first crop", "", "```", json.dumps(coverage), "```", "",
               table(e2, [m for m in E2_METRICS if m in e2], "layer").to_markdown(), "",
               pd.DataFrame(rep["e2_contrasts"]).round(3).to_markdown()]
    Path(a.out + ".md").write_text("\n".join(md))
    Path(a.out + ".json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print("wrote", a.out + ".md")


if __name__ == "__main__":
    main()
