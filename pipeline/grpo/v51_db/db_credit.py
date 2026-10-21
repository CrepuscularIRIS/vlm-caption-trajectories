"""Decision-state baselines for follow-up tool turns (idea run tool-credit-assignment, attempt 2).

Objective unchanged: the mentor's clean RLOO on r_i = correct_i (1 - lambda N_i). Only which tokens receive which part
of the outcome changes:
  turn 1 (shared group state)        every generated token keeps the RLOO advantage A_i
  turn k >= 2 (decision state s_k)   every generated token gets c_ik = r_i - b(s_k), whatever WORD was sampled there
                                     (crop word, commit, natural answer turn)
  observation tokens                 masked (unchanged)
b(s_k) = (1 - lambda (k - 1)) * mean_j z_kj: the policy's own answer-now value at s_k, from m_now forks of the same
policy with the prefill '<think>\\nStep k | commit\\n', z_kj = 1 iff the fork parses to an answer equal to gold and is not
truncated. b carries no gradient and depends only on the prefix of turn k (turns 1..k-1 and fork noise), so it is a
causal, action-independent baseline (unit test: test_db_credit.py).
CAPTION_DB_MODE: db (own state's b) | perm (negative control: b of a uniformly drawn other rollout of the same item at the
same index k; fallback: leave-one-out mean return of the item's other rollouts that reached k, then of the item) | croponly (attempt 1's rule, kept as a probe arm: b(s_k)
on turn k only when turn k sampled a crop; biased by -pi(1-pi)(V_stop - J), measured not hidden) | rloo (identity).
"""
from __future__ import annotations

import os
import random

MODES = ("db", "perm", "croponly", "rloo")


def mode() -> str:
    m = os.environ.get("CAPTION_DB_MODE", "db")
    if m not in MODES:
        raise ValueError(f"CAPTION_DB_MODE={m!r} not in {MODES}")
    return m


def segments(mask_row) -> list:
    """[start, end) of every contiguous run of 1s (one per assistant turn)."""
    out, s, n = [], None, len(mask_row)
    for t in range(n):
        if mask_row[t] and s is None:
            s = t
        elif not mask_row[t] and s is not None:
            out.append((s, t))
            s = None
    if s is not None:
        out.append((s, n))
    return out


def fork_value(z: list, k: int, lam: float) -> float | None:
    """b(s_k) from fork outcomes z (0/1); None when no fork could be run at s_k."""
    if not z:
        return None
    return (1.0 - lam * (k - 1)) * sum(z) / len(z)


def perm_baselines(b_lists: list, groups: list, rewards: list, rng: random.Random) -> list:
    """Negative control (audit revision 2): for every (rollout i, turn index t >= 1) with an own b, draw b from ANOTHER
    rollout of the SAME item (group) at the SAME decision index t. Fallback when no other rollout of the item reached
    index t with a b: the leave-one-out mean return of the other rollouts of the item that reached index t; else the
    leave-one-out mean return of the item."""
    pool, reach = {}, {}
    for i, bl in enumerate(b_lists):
        for t, b in enumerate(bl):
            if t >= 1:
                reach.setdefault((groups[i], t), []).append(i)
                if b is not None:
                    pool.setdefault((groups[i], t), []).append((i, b))
    tot, cnt = {}, {}
    for g, r in zip(groups, rewards):
        tot[g] = tot.get(g, 0.0) + r
        cnt[g] = cnt.get(g, 0) + 1
    out = []
    for i, bl in enumerate(b_lists):
        row = []
        g = groups[i]
        for t, b in enumerate(bl):
            if t == 0 or b is None:
                row.append(None)
                continue
            others = [bb for j, bb in pool.get((g, t), []) if j != i]
            if others:
                row.append(rng.choice(others))
                continue
            rr = [rewards[j] for j in reach.get((g, t), []) if j != i]
            row.append(sum(rr) / len(rr) if rr else (tot[g] - rewards[i]) / max(1, cnt[g] - 1))
        out.append(row)
    return out


def apply_db(adv, mask, rewards: list, b_lists: list, groups: list, m: str, rng: random.Random | None = None,
             kinds: list | None = None):
    """adv, mask: (B, L) tensors (adv = RLOO token advantage); rewards[i] = r_i; b_lists[i][t] = b of turn t+1 or None;
    kinds[i][t] = parsed action kind of turn t+1 (needed for croponly). Returns (new advantage, tokens rewritten)."""
    if m == "rloo":
        return adv, 0
    if m == "croponly":
        if kinds is None:
            raise ValueError("croponly needs per-turn action kinds")
        b_lists = [[b if (t < len(kinds[i]) and kinds[i][t] == "grounding") else None for t, b in enumerate(bl)]
                   for i, bl in enumerate(b_lists)]
    if m == "perm":
        b_lists = perm_baselines(b_lists, groups, rewards, rng or random.Random(0))
    out = adv.clone()
    rows = mask.detach().cpu().tolist()
    changed = 0
    for i, bl in enumerate(b_lists):
        segs = segments(rows[i])
        for t, b in enumerate(bl):
            if t == 0 or b is None or t >= len(segs):
                continue
            s, e = segs[t]
            out[i, s:e] = float(rewards[i]) - float(b)
            changed += e - s
    return out, changed


_STEP = 0


def install_advantage_patch():
    import verl.trainer.ray_trainer as rt
    if getattr(rt.compute_advantage, "_db_patched", False):
        return
    original = rt.compute_advantage

    def compute_advantage(data, *args, **kwargs):
        global _STEP
        data = original(data, *args, **kwargs)
        mm = data.non_tensor_batch.get("multi_modal_data")
        if mm is None or not len(mm) or "caption_b" not in mm[0]:
            return data
        _STEP += 1
        md = mode()
        b_lists = [list(x.get("caption_b") or []) for x in mm]
        groups = [str(g) for g in data.non_tensor_batch["uid"]]
        rewards = [float(x) for x in data.batch["token_level_rewards"].sum(dim=-1).tolist()]
        old = data.batch["advantages"]
        rng = random.Random(1_000_003 * _STEP + 17)
        kinds = [list(x.get("caption_kinds") or []) for x in mm]
        new, changed = apply_db(old, data.batch["response_mask"], rewards, b_lists, groups, md, rng, kinds)
        new = new.to(old.dtype)
        data.batch["advantages"] = new
        data.batch["returns"] = data.batch["returns"] + (new - old)
        # second moment of follow-up credit vs RLOO on the same tokens
        rows = data.batch["response_mask"].detach().cpu().tolist()
        c2_new = c2_old = n = 0.0
        for i, bl in enumerate(b_lists):
            segs = segments(rows[i])
            for t in range(1, min(len(bl), len(segs))):
                if bl[t] is None:
                    continue
                s = segs[t][0]
                c2_new += float(new[i, s]) ** 2
                c2_old += float(old[i, s]) ** 2
                n += 1
        nb = sum(1 for bl in b_lists for t, b in enumerate(bl) if t >= 1 and b is not None)
        print(f"[db_credit] mode={md} step={_STEP} followup_turns={nb} tokens_changed={changed} "
              f"Ec2_new={c2_new / max(1, n):.4f} Ec2_rloo={c2_old / max(1, n):.4f}", flush=True)
        return data

    compute_advantage._db_patched = True
    rt.compute_advantage = compute_advantage
