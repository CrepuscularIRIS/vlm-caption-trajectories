"""GRPO reward: terminal correctness x crop cost, correct * (1.0 - LAMBDA * crops).

LAMBDA comes from the required env CAPTION_CROP_LAMBDA (crop-cost round 2; the
advantage std floor that makes LAMBDA meaningful lives in easyr1_caption.install).

correct keeps the original terminal-correctness logic: exactly one <answer> whose
letter equals gold, else 0. crops is the environment-executed crop count for this
trajectory (budget 8), read from extra_info["crops"]. Wrong answers
stay 0, so the cost only distinguishes among correct samples.
No format bonus, abstention bonus, label bonus, or process reward.
Input must contain generated assistant tokens, not observation receipts.
"""
import os
import re

REWARD_NAME = "caption_correctness_cropfloor"
CROP_LAMBDA = float(os.environ["CAPTION_CROP_LAMBDA"])  # KeyError if unset: no silent default
REWARD_TYPE = "batch"
_ANS = re.compile(r"<answer>\s*([^<>]*?)\s*</answer>", re.S)
_LETTER = re.compile(r"[A-J]")


def _norm_answer(text: str) -> str | None:
    matches = _ANS.findall(text)
    # A trajectory terminates on its first answer; multiple answers are invalid.
    if len(matches) != 1:
        return None
    answer = matches[0].strip().upper()
    return answer if _LETTER.fullmatch(answer) else None


def correct_one(data_source: str, solution_str: str, ground_truth: str) -> float:
    """Undiscounted terminal correctness (the original reward)."""
    gold = str(ground_truth).strip().upper()
    return float(bool(_LETTER.fullmatch(gold)) and _norm_answer(solution_str) == gold)


def score_one(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    correct = correct_one(data_source, solution_str, ground_truth)
    crops = 0
    if isinstance(extra_info, dict):
        crops = int(extra_info.get("crops", 0) or 0)
    return correct * (1.0 - CROP_LAMBDA * crops)


def compute_score(reward_inputs: list[dict]) -> list[float]:
    rewards = []
    for item in reward_inputs:
        gold = item.get("ground_truth", item.get("reward_model", {}))
        if isinstance(gold, dict):
            gold = gold.get("ground_truth", "")
        rewards.append(score_one(item.get("data_source", "caption"), item.get("response", ""), str(gold), item))
    return rewards
