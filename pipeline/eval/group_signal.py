"""Where does the GRPO signal come from? Group composition and advantage shares from per-rollout logs.

Usage: python group_signal.py --log outputs/grpo_v3_A_cropcost_rollouts.jsonl --lam 0.01 [--steps 1-24]
For each step range it reports, under the logged reward r = correct * (1 - lam * crops):
  - share of groups that are all-correct / mixed / all-wrong (by correctness);
  - share of sum |advantage| that comes from all-correct groups under plain GRPO (std + 1e-6);
  - net advantage summed over direct (crops == 0) samples, split by group type;
  - the same under DAPO-style filtering on accuracy (only mixed groups kept), with lam and with lam = 0,
    so the crop term's share inside mixed groups is visible.
Text output only.
"""
import argparse
import collections
import json

import numpy as np

EPS = 1e-6


def advantages(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean()) / (scores.std() + EPS)


def load(path: str) -> dict[int, dict[str, list[dict]]]:
    by_step: dict[int, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_step[r["step"]][r["uid"]].append(r)
    return by_step


def summarize(groups: list[list[dict]], lam: float) -> dict[str, float]:
    kinds = collections.Counter()
    abs_adv = collections.Counter()
    direct_net = collections.Counter()
    ds_direct_net = ds0_direct_net = crop_term = ds_abs = 0.0
    n_rows = n_direct = 0
    for g in groups:
        corr = np.array([r["correct"] for r in g], dtype=float)
        crops = np.array([r["crops"] or 0 for r in g], dtype=float)
        direct = crops == 0
        n_rows += len(g)
        n_direct += int(direct.sum())
        kind = "all_correct" if corr.min() == 1 else "all_wrong" if corr.max() == 0 else "mixed"
        kinds[kind] += 1
        adv = advantages(corr * (1 - lam * crops))
        abs_adv[kind] += np.abs(adv).sum()
        direct_net[kind] += adv[direct].sum()
        if kind == "mixed":  # the only groups DAPO filtering on accuracy keeps
            adv0 = advantages(corr)
            ds_direct_net += adv[direct].sum()
            ds0_direct_net += adv0[direct].sum()
            crop_term += np.abs(adv - adv0).sum()
            ds_abs += np.abs(adv).sum()
    n = sum(kinds.values())
    tot = sum(abs_adv.values()) or 1.0
    return dict(groups=n, rows=n_rows, direct_rate=n_direct / max(n_rows, 1),
                all_correct=kinds["all_correct"] / n, mixed=kinds["mixed"] / n, all_wrong=kinds["all_wrong"] / n,
                absadv_share_all_correct=abs_adv["all_correct"] / tot,
                direct_net_all_correct=direct_net["all_correct"], direct_net_mixed=direct_net["mixed"],
                ds_direct_net_lam=ds_direct_net, ds_direct_net_lam0=ds0_direct_net,
                ds_crop_term_share=crop_term / (ds_abs or 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--ranges", nargs="+", default=["1-8", "9-16", "17-24", "25-40", "41-56"])
    a = ap.parse_args()
    by_step = load(a.log)
    print(f"# {a.log}  lam={a.lam}  steps logged {min(by_step)}-{max(by_step)}")
    for rng in a.ranges:
        lo, hi = map(int, rng.split("-"))
        groups = [g for s in range(lo, hi + 1) for g in by_step.get(s, {}).values()]
        if not groups:
            continue
        s = summarize(groups, a.lam)
        print(f"steps {rng}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))


if __name__ == "__main__":
    main()
