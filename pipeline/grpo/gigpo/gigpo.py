"""GiGPO for the caption crop agent (class-4 baseline, mentor 10-27; settings obs/notes/mainline/11 §3).

A = episode advantage + w * step advantage (w = CAPTION_GIGPO_STEP_W, default 1.0).
- episode: EasyR1 GRPO outcome advantage per uid (mean/std over the 16 rollouts), broadcast to every generated token.
- step: for every assistant turn, its anchor = (uid, admitted views, executed crop boxes, crops used, step index) recorded
  by the rollout before the turn was sampled. Anchors equal across rollouts form a step group; the step return with
  gamma = 1 and a terminal-only reward is the trajectory reward R; step advantage = (R - mean_g) / (std_g + eps) over the
  group (mean_std_norm), 0 for singleton groups, added to that turn's generated tokens only (receipts carry no loss).
Active when CAPTION_GIGPO=1 (driver-side replacement of ray_trainer.compute_advantage); otherwise plain EasyR1.
"""
from __future__ import annotations

import os
from collections import defaultdict

import torch


def step_w() -> float:
    return float(os.environ.get("CAPTION_GIGPO_STEP_W", "1.0"))


def step_advantages(uids, scores: list[float], anchors: list[list], eps: float = 1e-6):
    """Per rollout, per turn step advantage, plus group statistics. Pure (CPU-testable)."""
    groups = defaultdict(list)
    for i, (u, turns) in enumerate(zip(uids, anchors)):
        for t, a in enumerate(turns):
            groups[(u, tuple(a))].append((i, t))
    out = [[0.0] * len(turns) for turns in anchors]
    stats = {"root_groups": 0, "root_multi": 0, "nonroot_groups": 0, "nonroot_multi": 0, "turns": 0, "nonzero": 0}
    for (u, a), members in groups.items():
        root = a[3] == 0
        stats["root_groups" if root else "nonroot_groups"] += 1
        if len(members) < 2:
            continue
        stats["root_multi" if root else "nonroot_multi"] += 1
        r = torch.tensor([scores[i] for i, _ in members], dtype=torch.float64)
        adv = (r - r.mean()) / (r.std() + eps)
        for (i, t), v in zip(members, adv.tolist()):
            out[i][t] = v
    for row in out:
        stats["turns"] += len(row)
        stats["nonzero"] += sum(v != 0.0 for v in row)
    return out, stats


def install() -> None:
    if os.environ.get("CAPTION_GIGPO", "0") != "1":
        return
    import verl.trainer.ray_trainer as RT
    if getattr(RT.compute_advantage, "_gigpo", False):
        return
    original = RT.compute_advantage

    def compute_advantage(data, adv_estimator, gamma: float = 1.0, lam: float = 1.0):
        if gamma != 1.0:
            raise RuntimeError("GiGPO port assumes gamma = 1 (terminal-only reward)")
        data = original(data, adv_estimator, gamma=gamma, lam=lam)  # episode part: EasyR1 GRPO (adv_estimator=grpo)
        mm = data.non_tensor_batch["multi_modal_data"]
        scores = data.batch["token_level_rewards"].sum(-1).tolist()
        steps, st = step_advantages(list(data.non_tensor_batch["uid"]), scores, [m["caption_anchors"] for m in mm])
        adv = data.batch["advantages"]
        mask = data.batch["response_mask"]
        w = step_w()
        for i, m in enumerate(mm):
            for (a, b), v in zip(m["caption_turn_spans"], steps[i]):
                if v != 0.0:
                    adv[i, a:b] += w * v * mask[i, a:b]
        data.batch["advantages"] = adv
        data.batch["returns"] = adv
        print(f"[gigpo] w={w} root_groups={st['root_groups']} root_multi={st['root_multi']} "
              f"nonroot_groups={st['nonroot_groups']} nonroot_multi={st['nonroot_multi']} "
              f"turns={st['turns']} step_adv_nonzero={st['nonzero']}", flush=True)
        return data

    compute_advantage._gigpo = True
    RT.compute_advantage = compute_advantage
