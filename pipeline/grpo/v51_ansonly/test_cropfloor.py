"""CPU tests for crop-cost round 2: reward correct * (1 - LAMBDA * crops) and the std-floored GRPO advantage.

Run: CAPTION_CROP_LAMBDA=0.05 python test_cropfloor.py  (exit 0 = all pass)
"""
import os
import sys
from collections import defaultdict

import torch

assert os.environ.get("CAPTION_CROP_LAMBDA") == "0.05", "run with CAPTION_CROP_LAMBDA=0.05"
from reward_caption import score_one  # noqa: E402

DIRECT = "The chart shows 2.1 for Qtr2. <answer>A</answer>"
WRONG = "Maybe it is B. <answer>B</answer>"


def plain_grpo(scores, index, eps=1e-6):
    groups = defaultdict(list)
    for s, i in zip(scores, index):
        groups[i].append(s)
    stats = {i: (torch.mean(torch.tensor(g)), torch.std(torch.tensor(g))) for i, g in groups.items()}
    return torch.stack([(s - stats[i][0]) / (stats[i][1] + eps) for s, i in zip(scores, index)])


def floored(scores, index, floor=0.25):
    from easyr1_caption import grpo_std_floor_advantage
    rewards = torch.zeros(len(scores), 4)
    rewards[:, -1] = torch.tensor(scores)
    adv, _ = grpo_std_floor_advantage(rewards, torch.ones(len(scores), 4), index, floor)
    return adv[:, 0]


def main() -> int:
    ok = True

    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(float(got) - want) < tol
        ok &= good
        print(("PASS" if good else "FAIL"), name, float(got), want)

    check("reward direct correct", score_one("caption", DIRECT, "A", {"crops": 0}), 1.0)
    check("reward 2 crops correct", score_one("caption", DIRECT, "A", {"crops": 2}), 0.90)
    check("reward wrong", score_one("caption", WRONG, "A", {"crops": 0}), 0.0)
    check("reward invalid (empty)", score_one("caption", "", "A", {"crops": 0}), 0.0)

    idx = ["q"] * 16
    s = [1.0] * 8 + [0.95] * 8                      # all correct, crops 0 vs 1, lambda .05
    a = floored(s, idx)
    check("all-correct 0/1 crops: direct adv", a[0], 0.1, 1e-4)
    check("all-correct 0/1 crops: crop adv", a[-1], -0.1, 1e-4)
    a = floored([0.9] * 16, idx)
    check("all-correct same crops -> 0", a.abs().max(), 0.0)

    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(200):                             # lambda = 0: identical to plain GRPO
        s = torch.randint(0, 2, (48,), generator=g).float().tolist()
        index = [f"q{i // 16}" for i in range(48)]
        worst = max(worst, float((floored(s, index) - plain_grpo(s, index)).abs().max()))
    check("lambda=0 matches plain GRPO (max diff < 1e-4)", worst < 1e-4, 1.0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
