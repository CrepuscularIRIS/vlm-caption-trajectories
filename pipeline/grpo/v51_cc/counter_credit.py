"""CounterCredit price channel for caption + crop GRPO (card A / card B, 2026-10-18).

Follows CounterCredit (arXiv 2609.22910) §3 and App. C/E, adapted to caption protocol v5.1:
- h_i  = rollout ids up to and including the assistant's </think> of the turn that issued crop i
         (the caption is part of the pre-call state, as in the paper).
- probe = append the elicitation E as a user turn (merged into the receipt's user turn when the
          prefix already ends in one), open the assistant turn and prefill "<answer"; read the
          next-token distribution over ">A" ... ">J" (Qwen tokenizer merges ">" with the letter).
- U     = p(gold letter) renormalised over the ten letters (letters outside the top-20 logprobs
          count as 0). Paper renormalises over the question's own options; ours use A-J.
- D = U_real - U_now, E = U_real - mean_k U_rand_k, q = min(D, E).
- verified = trajectory correct AND eligible AND q > eps; price
  P = min(C, gamma * sum_verified (q - eps)) - rho * (crops - n_verified).
- advantage = GRPO(correctness) + (P - group mean P); price advantage 0 in groups with no correct
  rollout. Outcome channel is EasyR1's unmodified GRPO (with G=16 a mixed group has std >= 0.25).
Constants are env-overridable so the null calibration can set eps before the real runs.
"""
from __future__ import annotations

import math
import os
import random
from collections import defaultdict

ELICIT = "Based on everything so far, answer the original question now. Reply with the option letter only."
LETTERS = "ABCDEFGHIJ"
THINK_END_ID = 151668  # "</think>" in the Qwen3-VL tokenizer (checked on the SFT-A checkpoint)


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


RHO = _f("CC_RHO", 0.05)
GAMMA = _f("CC_GAMMA", 0.5)
EPS = _f("CC_EPS", 0.05)
K = int(_f("CC_K", 3))
N_MAX = 8                      # caption_env.MAX_CROPS
CAP = _f("CC_CAP", 1.0 - RHO * N_MAX)   # paper: C = 1 - rho * n_max (0.6 here)


def letter_token_ids(tokenizer) -> list[int]:
    ids = []
    for letter in LETTERS:
        enc = tokenizer.encode(">" + letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"'>{letter}' is not a single token: {enc}")
        ids.append(enc[0])
    return ids


def gold_letter(gold) -> str | None:
    if isinstance(gold, dict):
        gold = gold.get("ground_truth", "")
    gold = str(gold).strip().upper()
    return gold if len(gold) == 1 and gold in LETTERS else None


def score_from_logprobs(top: dict, letter_ids: list[int], gold: str) -> float | None:
    """top: {token_id: logprob float}. Returns U or None when no letter has mass."""
    probs = [math.exp(top[t]) if t in top else 0.0 for t in letter_ids]
    total = sum(probs)
    if total <= 0.0:
        return None
    return probs[LETTERS.index(gold)] / total


def now_suffix() -> str:
    return "<|im_end|>\n<|im_start|>user\n" + ELICIT + "<|im_end|>\n<|im_start|>assistant\n<answer"


def receipt_probe_chunk(receipt: dict, close: str) -> str:
    return (close + "\n<|im_start|>user\n" + receipt["head"] +
            "\n<|vision_start|><|image_pad|><|vision_end|>" + receipt["tail"] +
            "\n" + ELICIT + "<|im_end|>\n<|im_start|>assistant\n<answer")


def random_patch(env, box, rng: random.Random):
    """Same-size patch of the full-resolution original at a uniform random position."""
    from caption_env import present
    w = int(box[2]) - int(box[0])
    h = int(box[3]) - int(box[1])
    x0 = rng.randint(0, max(0, env.W - w))
    y0 = rng.randint(0, max(0, env.H - h))
    img, _ = present(env.full.crop((x0, y0, x0 + w, y0 + h)), is_original=False)
    return img


def crop_values(u_now: float | None, u_real: float | None, u_rand: list) -> dict:
    u_rand = [u for u in u_rand if u is not None]
    if u_now is None or u_real is None or not u_rand:
        return {"eligible": False}
    d = u_real - u_now
    e = u_real - sum(u_rand) / len(u_rand)
    return {"eligible": True, "u_now": u_now, "u_real": u_real, "u_rand": u_rand,
            "D": d, "E": e, "q": min(d, e)}


def trajectory_price(correct: bool, crops: int, values: list[dict],
                     rho: float = None, gamma: float = None, eps: float = None, cap: float = None) -> dict:
    rho = RHO if rho is None else rho
    gamma = GAMMA if gamma is None else gamma
    eps = EPS if eps is None else eps
    cap = CAP if cap is None else cap
    verified = [v for v in values if correct and v.get("eligible") and v["q"] > eps]
    cashback = min(cap, gamma * sum(v["q"] - eps for v in verified))
    price = cashback - rho * (crops - len(verified))
    return {"price": price, "n_verified": len(verified), "cashback": cashback}


def price_advantage(uids, prices, corrects) -> list[float]:
    """Centered price per group; zero when the group has no correct rollout."""
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    out = [0.0] * len(uids)
    for idx in groups.values():
        if not any(corrects[i] > 0.5 for i in idx):
            continue
        mean = sum(prices[i] for i in idx) / len(idx)
        for i in idx:
            out[i] = prices[i] - mean
    return out


def install_advantage_patch():
    """Wrap ray_trainer.compute_advantage: GRPO outcome advantage + centered price channel."""
    import torch
    import verl.trainer.ray_trainer as rt
    if getattr(rt.compute_advantage, "_cc_patched", False):
        return
    original = rt.compute_advantage

    def compute_advantage(data, *args, **kwargs):
        data = original(data, *args, **kwargs)
        mm = data.non_tensor_batch.get("multi_modal_data")
        if mm is None or not len(mm) or "caption_price" not in mm[0]:
            return data
        uids = [str(u) for u in data.non_tensor_batch["uid"]]
        prices = [float(m["caption_price"]) for m in mm]
        corrects = [float(m.get("caption_correct", 0.0)) for m in mm]
        adv = price_advantage(uids, prices, corrects)
        mask = data.batch["response_mask"].to(data.batch["advantages"].dtype)
        delta = torch.tensor(adv, dtype=mask.dtype, device=mask.device).unsqueeze(-1) * mask
        data.batch["advantages"] = data.batch["advantages"] + delta
        data.batch["returns"] = data.batch["returns"] + delta
        n = len(prices)
        print(f"[counter_credit] price_mean={sum(prices)/n:.4f} "
              f"price_adv_abs_mean={sum(abs(a) for a in adv)/n:.4f} "
              f"verified={sum(int(m.get('caption_verified', 0)) for m in mm)} "
              f"crops={sum(int(m.get('caption_crops', 0)) for m in mm)}", flush=True)
        return data

    compute_advantage._cc_patched = True
    rt.compute_advantage = compute_advantage
