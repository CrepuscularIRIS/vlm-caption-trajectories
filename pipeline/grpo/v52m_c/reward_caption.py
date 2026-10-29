"""GRPO terminal correctness ONLY: correct=1, every other outcome=0.

No format bonus, abstention bonus, crop bonus, or process reward.
Input must contain generated assistant tokens, not observation receipts.
"""
import re

REWARD_NAME = "caption_correctness"
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


def score_one(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    gold = str(ground_truth).strip().upper()
    return float(bool(_LETTER.fullmatch(gold)) and _norm_answer(solution_str) == gold)


def compute_score(reward_inputs: list[dict]) -> list[float]:
    rewards = []
    for item in reward_inputs:
        gold = item.get("ground_truth", item.get("reward_model", {}))
        if isinstance(gold, dict):
            gold = gold.get("ground_truth", "")
        rewards.append(score_one(item.get("data_source", "caption"), item.get("response", ""), str(gold)))
    return rewards
