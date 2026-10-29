"""B6 (GRPO without tool) reward: 1 if the last <answer>X</answer> equals the gold letter, else 0.

Same correctness-only rule as reward_caption; no format bonus. Prompt = eval_base direct mode.
"""
import re
from typing import Any

REWARD_NAME = "direct_correctness"
REWARD_TYPE = "sequential"
ANS_RE = re.compile(r"<answer>\s*\(?\s*([A-Z])\s*\)?\s*</answer>", re.I)


def compute_score(reward_input: dict[str, Any]) -> dict[str, float]:
    m = ANS_RE.findall(reward_input["response"])
    gold = reward_input["ground_truth"]
    gold = gold.get("ground_truth", "") if isinstance(gold, dict) else str(gold)
    ok = float(bool(m) and m[-1].upper() == gold.strip().upper())
    return {"overall": ok, "accuracy": ok}
