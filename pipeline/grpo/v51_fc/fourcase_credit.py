"""Four-case tool credit with the policy's own before/after values (mentor 10-21; replaces the two-cell tool_credit.py).

Objective and estimator unchanged: clean RLOO on r_i = correct_i (1 - lambda N_i). Only the tokens of crop-issuing turns
get a different signal:
  v(s_t)    answer-now value of the policy at decision state s_t (before turn t is sampled): the share of forks of the
            same policy, prefilled '<think>\\nStep t | commit\\n', that end in a parsable, untruncated answer equal to gold.
            s_1 is shared by all rollouts of a question; its forks are pooled across the group.
  crop turn t (its caption, header and <grounding> action) gets c = v(s_{t+1}) - v(s_t) - lambda, i.e. the change in
            the policy's chance to answer right now caused by this crop and its receipt, minus the crop price. With binary
            before/after labels this is the mentor's table: WR +1, RR 0, WW 0, RW -1 (each minus lambda); with fork means it
            is the continuous version.
  every other turn (answer turns, invalid turns) and every crop turn whose before or after value is missing keeps A.
Labels for reporting only: before / after = 'R' if v >= 0.5 else 'W'.
CAPTION_FC_MODE: fourcase | rloo (identity; parity smoke).
"""
from __future__ import annotations

import os

MODES = ("fourcase", "rloo")
CASES = ("WR", "RR", "WW", "RW")


def mode() -> str:
    m = os.environ.get("CAPTION_FC_MODE", "fourcase")
    if m not in MODES:
        raise ValueError(f"CAPTION_FC_MODE={m!r} not in {MODES}")
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


def fork_value(z: list) -> float | None:
    return sum(z) / len(z) if z else None


def crop_credit(v_before: float, v_after: float, lam: float) -> float:
    return v_after - v_before - lam


def case_label(v_before: float, v_after: float) -> str:
    return ("R" if v_before >= 0.5 else "W") + ("R" if v_after >= 0.5 else "W")


def apply_fourcase(adv, mask, v_lists: list, kinds: list, lam: float, m: str):
    """adv, mask: (B, L) tensors (adv = RLOO token advantage); v_lists[i][t] = v(s_{t+1}) of rollout i (0-based turn t);
    kinds[i][t] = parsed action kind of turn t. Returns (new advantage, tokens rewritten, stats)."""
    stats = {c: 0 for c in CASES}
    stats.update(crop_turns=0, credited=0, sum_c=0.0, sum_a=0.0)
    if m == "rloo":
        return adv, 0, stats
    out = adv.clone()
    rows = mask.detach().cpu().tolist()
    changed = 0
    for i, kl in enumerate(kinds):
        segs = segments(rows[i])
        v = v_lists[i]
        for t, kind in enumerate(kl):
            if kind != "grounding":
                continue
            stats["crop_turns"] += 1
            if t >= len(segs) or t + 1 >= len(v) or v[t] is None or v[t + 1] is None:
                continue
            s, e = segs[t]
            c = crop_credit(v[t], v[t + 1], lam)
            stats["sum_a"] += float(adv[i, s])
            out[i, s:e] = c
            changed += e - s
            stats["credited"] += 1
            stats["sum_c"] += c
            stats[case_label(v[t], v[t + 1])] += 1
    return out, changed, stats


def install_advantage_patch():
    import verl.trainer.ray_trainer as rt
    if getattr(rt.compute_advantage, "_fc_patched", False):
        return
    original = rt.compute_advantage

    def compute_advantage(data, *args, **kwargs):
        data = original(data, *args, **kwargs)
        mm = data.non_tensor_batch.get("multi_modal_data")
        if mm is None or not len(mm) or "caption_v" not in mm[0]:
            return data
        md = mode()
        lam = float(os.environ["CAPTION_CROP_LAMBDA"])
        v_lists = [list(x.get("caption_v") or []) for x in mm]
        kinds = [list(x.get("caption_kinds") or []) for x in mm]
        old = data.batch["advantages"]
        new, changed, st = apply_fourcase(old, data.batch["response_mask"], v_lists, kinds, lam, md)
        new = new.to(old.dtype)
        data.batch["advantages"] = new
        data.batch["returns"] = data.batch["returns"] + (new - old)
        k = max(1, st["credited"])
        print(f"[fc_credit] mode={md} crop_turns={st['crop_turns']} credited={st['credited']} tokens_changed={changed} "
              f"WR={st['WR']} RR={st['RR']} WW={st['WW']} RW={st['RW']} mean_c={st['sum_c'] / k:.4f} "
              f"mean_A_replaced={st['sum_a'] / k:.4f}", flush=True)
        return data

    compute_advantage._fc_patched = True
    rt.compute_advantage = compute_advantage
