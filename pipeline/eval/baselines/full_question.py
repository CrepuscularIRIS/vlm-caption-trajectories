"""Recover the complete user question from the existing caption env_head.

Pure function: receives only env_head, never reward_model/answer/ground_truth.
No API/model/filesystem calls. It removes only the protocol wrapper and the
terminal image metadata, preserving the full question and options verbatim.
"""
import re

_SUFFIX = re.compile(r"\nOriginal image \[original_image\]\. Source pixels: [0-9]+x[0-9]+, [^\n]+\nImage \[original_image\]:\Z")


def full_question(env_head: str) -> str:
    if not isinstance(env_head, str) or not env_head.startswith("Question:"):
        raise ValueError("Expected caption env_head with Question: prefix")
    suffix = _SUFFIX.search(env_head)
    if suffix is None:
        raise ValueError("Expected terminal original-image metadata in env_head")
    question = env_head[len("Question:"):suffix.start()]
    # Protocol emits one separating space for nonempty question text; a question
    # with only image-based content may begin directly with a newline + options.
    if question.startswith(" "):
        question = question[1:]
    if not question.strip():
        raise ValueError("Empty question and options")
    return question
