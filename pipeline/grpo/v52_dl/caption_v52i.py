"""Protocol v5.2i = v5.1 with ONE change (Pro design 10-22 §1.2): Step 1's WORD records an intent only.

v5.1 binds the header WORD to the action (locate/re_examine/fallback -> <grounding>, commit/abstain -> <answer>,
abstain -> UNCLEAR). Under that binding the same post-caption prefix h cannot legally take both actions, so the
answer-vs-crop comparison at h would measure format legality, not task value. Here, for expected_step == 1 only, the
three WORD-derived errors are dropped; the real action is the final <answer> or <grounding>. Format, step number,
sources, citations, bbox geometry, budget and the answer rule are unchanged; later steps keep v5.1 exactly.

SYSTEM text is NOT changed (the SFT-A policy was trained on it), so the dataset SYSTEM assertion still holds. Main
rollouts, forks, rewards and evaluation all import this module through caption_env.proto, so they share one version.
"""
from __future__ import annotations

from caption_v51 import *  # noqa: F401,F403  (SYSTEM, parse_step, first_user_text, receipt_text, ...)
from caption_v51 import MAX_CROPS
from caption_v51 import check_step as _check_step_v51

PROTOCOL = "v5.2i"
STEP1_RELAXED = ("behavior_requires_grounding", "behavior_requires_answer", "abstain_without_UNCLEAR")


def check_step(p: dict, expected_step: int, crops_used: int, available: list[str], budget: int = MAX_CROPS) -> list[str]:
    errors = _check_step_v51(p, expected_step, crops_used, available, budget)
    if expected_step != 1:
        return errors
    return [e for e in errors if e not in STEP1_RELAXED]
