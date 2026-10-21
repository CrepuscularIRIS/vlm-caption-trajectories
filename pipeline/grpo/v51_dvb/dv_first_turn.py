"""First-turn two-branch value term (mentor 10-19): only the tokens of the FIRST assistant turn get an extra advantage
s_i * dV(q): s_i = +1 if the rollout's first action is a crop (grounding), -1 if it answers directly, 0 if the first
turn is invalid / truncated. dV(q) = V_crop(q) - V_direct(q) is precomputed once at the fixed library-A SFT checkpoint
(two branches x 4 samples, identical prompt; user rule: branch value = mean over met samples, dV = 0 when either
branch has < 2 met samples). Weight 1, not group-centred. All later turns keep the original GRPO advantage.

CAPTION_DV_MODE selects how the term enters the advantage (docs/29 section 6.3 item 1, direction B):
  add_first             (default, the 10-20 dV arm) first-turn tokens: A_i + s_i * dV
  replace_first_centred first-turn tokens: c_i = s_i * dV - mean_group(s * dV) replaces A_i (no outcome advantage on the
                        first turn); later turns keep A_i
  add_all_centred       every response token: A_i + c_i
The group is EasyR1's GRPO group (non_tensor_batch['uid'], the 16 rollouts of one prompt). c_i is 0 in a group whose
rollouts all took the same first action.
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


MODES = ("add_first", "replace_first_centred", "add_all_centred")


def mode() -> str:
    m = os.environ.get("CAPTION_DV_MODE", "add_first")
    if m not in MODES:
        raise ValueError(f"CAPTION_DV_MODE={m!r} not in {MODES}")
    return m


def centred_terms(groups, dvs, kinds):
    """c_i = s_i * dV_i - mean over i's group of s * dV."""
    terms = [sign_of_first(k) * float(v) for k, v in zip(kinds, dvs)]
    tot, cnt = {}, {}
    for g, x in zip(groups, terms):
        tot[g] = tot.get(g, 0.0) + x
        cnt[g] = cnt.get(g, 0) + 1
    return [x - tot[g] / cnt[g] for g, x in zip(groups, terms)]


def apply_mode(adv, mask, groups, dvs, kinds, m):
    """New advantage tensor (B, L) for mode m; adv is the original GRPO token advantage."""
    import torch
    if m == "add_first":
        return adv + first_turn_delta(mask, dvs, kinds).to(adv.dtype)
    c = centred_terms(groups, dvs, kinds)
    out = adv.clone()
    rows = mask.detach().cpu().tolist()
    for i, row in enumerate(rows):
        if m == "replace_first_centred":
            s, e = first_segment(row)
            out[i, s:e] = c[i]
        else:
            out[i] = adv[i] + c[i] * mask[i].to(adv.dtype)
    return out * mask.to(adv.dtype) if m == "add_all_centred" else out


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
        groups = [str(g) for g in data.non_tensor_batch["uid"]]
        md = mode()
        old = data.batch["advantages"]
        new = apply_mode(old, mask, groups, dvs, kinds, md).to(old.dtype)
        data.batch["advantages"] = new
        data.batch["returns"] = data.batch["returns"] + (new - old)
        n = len(mm)
        nc = sum(k == "grounding" for k in kinds); nd = sum(k == "answer" for k in kinds)
        print(f"[dv_first] mode={md} n={n} first_crop={nc} first_direct={nd} other={n - nc - nd} "
              f"dv_nonzero={sum(abs(v) > 0 for v in dvs)} mean_abs_term={sum(abs(sign_of_first(k) * v) for k, v in zip(kinds, dvs)) / n:.4f} "
              f"mean_abs_centred={sum(abs(x) for x in centred_terms(groups, dvs, kinds)) / n:.4f}",
              flush=True)
        return data

    compute_advantage._dv_patched = True
    rt.compute_advantage = compute_advantage
