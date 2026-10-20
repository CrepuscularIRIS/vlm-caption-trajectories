"""First-turn two-branch value term (mentor 10-19): only the tokens of the FIRST assistant turn get an extra advantage
s_i * dV(q): s_i = +1 if the rollout's first action is a crop (grounding), -1 if it answers directly, 0 if the first
turn is invalid / truncated. dV(q) = V_crop(q) - V_direct(q) is precomputed once at the fixed library-A SFT checkpoint
(two branches x 4 samples, identical prompt; user rule: branch value = mean over met samples, dV = 0 when either
branch has < 2 met samples). Weight 1, not group-centred. All later turns keep the original GRPO advantage.
"""
import json
import os

_TABLE = None


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        path = os.environ["CAPTION_DV_TABLE"]  # KeyError if unset: no silent default
        _TABLE = {str(k): float(v) for k, v in json.load(open(path)).items()}
        need = int(os.environ.get("CAPTION_DV_EXPECT", "2000"))
        if len(_TABLE) != need:
            raise ValueError(f"dV table has {len(_TABLE)} entries, expected {need}")
    return _TABLE


MISSES = 0


def lookup(uid) -> float:
    """dV for a pool uid; validation / holdout uids are not in the training table and get 0 (counted in MISSES).
    The table must cover the whole 2,000-question training pool (checked once in table())."""
    global MISSES
    t = table()
    if str(uid) not in t:
        MISSES += 1
        return 0.0
    return t[str(uid)]


def sign_of_first(kind) -> int:
    return 1 if kind == "grounding" else -1 if kind == "answer" else 0


def first_segment(mask_row):
    """[start, end) of the first contiguous run of 1s in a response mask (the first assistant turn)."""
    n = len(mask_row)
    s = 0
    while s < n and not mask_row[s]:
        s += 1
    e = s
    while e < n and mask_row[e]:
        e += 1
    return s, e


def first_turn_delta(mask, dvs, kinds):
    """mask: (B, L) 0/1 tensor; returns a (B, L) tensor with s_i * dV_i on the first-turn tokens only."""
    import torch
    delta = torch.zeros_like(mask, dtype=torch.float32)
    rows = mask.detach().cpu().tolist()
    for i, row in enumerate(rows):
        v = sign_of_first(kinds[i]) * float(dvs[i])
        if v:
            s, e = first_segment(row)
            delta[i, s:e] = v
    return delta.to(mask.device)


def install_advantage_patch():
    import verl.trainer.ray_trainer as rt
    if getattr(rt.compute_advantage, "_dv_patched", False):
        return
    original = rt.compute_advantage

    def compute_advantage(data, *args, **kwargs):
        data = original(data, *args, **kwargs)
        mm = data.non_tensor_batch.get("multi_modal_data")
        if mm is None or not len(mm) or "caption_dv" not in mm[0]:
            return data
        mask = data.batch["response_mask"]
        dvs = [float(m["caption_dv"]) for m in mm]
        kinds = [m.get("caption_first") for m in mm]
        delta = first_turn_delta(mask, dvs, kinds).to(data.batch["advantages"].dtype)
        data.batch["advantages"] = data.batch["advantages"] + delta
        data.batch["returns"] = data.batch["returns"] + delta
        n = len(mm)
        nc = sum(k == "grounding" for k in kinds); nd = sum(k == "answer" for k in kinds)
        print(f"[dv_first] n={n} first_crop={nc} first_direct={nd} other={n - nc - nd} "
              f"dv_nonzero={sum(abs(v) > 0 for v in dvs)} mean_abs_term={sum(abs(sign_of_first(k) * v) for k, v in zip(kinds, dvs)) / n:.4f}",
              flush=True)
        return data

    compute_advantage._dv_patched = True
    rt.compute_advantage = compute_advantage
