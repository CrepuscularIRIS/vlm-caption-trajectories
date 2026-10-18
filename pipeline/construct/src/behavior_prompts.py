"""Step 1 of the mentor's recipe: behavior list -> generation instructions (v1, 2026-09-30).

Builds on the Mini-o3 App. F format (JSON per step, normalized grounding with a source image)
and adds an explicit `behavior` field. The four Mini-o3 exemplars are reused with a behavior
label on every step. HOLD / REVISE are not generated here; they are built later as paired
continuations of committed trajectories.
"""
import json

from minio3_prompts import EXEMPLARS, IMG

BEHAVIORS = {
    'answer': 'The answer is clearly readable in the ORIGINAL image without zooming: answer directly.',
    'locate': 'The target is not in view yet (or too small to see): choose a region to zoom into.',
    're_examine': 'The target is in view but not clear enough: zoom again on the same area with a new, better crop. '
                  'Never claim to look again without producing a new crop.',
    'compare': 'Several similar candidates could match: inspect them one at a time and state what tells them apart.',
    'fallback': 'The current view does not contain the target, or contradicts the question: say why, then go back to '
                'the original image or an earlier observation and search somewhere else.',
    'commit': 'The evidence is clearly visible: state exactly what you see and in which observation, then give the '
              'final answer.',
}

SYSTEM_V1 = """You answer questions about an image by looking carefully, step by step. You have two actions:
1. grounding([x1, y1, x2, y2], source): zoom into a region of the 'source' image. Coordinates are normalized to [0, 1]
   relative to that source image ((0, 0) is its top-left corner). 'source' is "original_image" or "observation_i"
   (one of the earlier zoomed views).
2. final(answer): give the final answer. For multiple-choice questions answer with the option letter.

At every step output ONE JSON object with three fields:
{"behavior": "...", "thought": "...", "action": "..."}

"behavior" must be one of:
""" + '\n'.join(f'- {k}: {v}' for k, v in BEHAVIORS.items()) + """

Rules:
- Only describe what is actually visible in the image you are looking at. If something is too small or blurry to read,
  say so instead of guessing.
- Use "answer" only at the first step and only when zooming is clearly unnecessary.
- Stop as soon as the evidence is sufficient; do not zoom further than the image resolution allows.

Here are several examples:"""

# Behavior labels for the Mini-o3 exemplars, step by step (same order as minio3_prompts.EXEMPLARS).
EXEMPLAR_BEHAVIORS = [
    ['locate', 're_examine', 'fallback', 'commit'],
    ['locate', 'locate', 're_examine', 'commit'],
    ['locate', 're_examine', 're_examine', 'commit'],
    ['locate', 'fallback', 'fallback', 'locate', 're_examine', 're_examine', 'commit'],
]

TURN_V1 = """Now output step {i} as ONE JSON object: {{"behavior": "...", "thought": "...", "action": "..."}}.
If the target is present in the latest view, decide whether the evidence is sufficient (commit) or needs a closer look
(re_examine). If it is not present, explain how the previous box missed it and where to look instead (fallback)."""


def exemplar_block_v1() -> str:
    parts = []
    for n, ((question, steps), labels) in enumerate(zip(EXEMPLARS, EXEMPLAR_BEHAVIORS), 1):
        lines = [f'Example {n}:', f'Question: {question}', f'Original Image: {IMG}']
        for i, ((thought, action), beh) in enumerate(zip(steps, labels)):
            step = json.dumps({'behavior': beh, 'thought': thought, 'action': action}, ensure_ascii=False)
            lines.append(f'Step {i}: {step}')
            if action.startswith('grounding'):
                lines.append(f'Observation {i + 1}: {IMG}')
        parts.append('\n'.join(lines))
    return '\n\n'.join(parts)


def system_prompt_v1() -> str:
    return SYSTEM_V1 + '\n\n' + exemplar_block_v1()


def turn_instruction_v1(i: int) -> str:
    return TURN_V1.format(i=i)
