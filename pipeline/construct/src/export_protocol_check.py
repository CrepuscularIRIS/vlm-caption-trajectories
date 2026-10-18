"""CPU protocol check on exported assistant turns (2026-10-06; Codex v0.1 review: 011/012 kept coordinates in captions
and 011 had text outside <think>, which the generation-time coordinate gate and the LLaMA-Factory check both missed).

v5.1 protocol per assistant turn: exactly one <think>...</think> block, then exactly one action (<grounding>{...}</grounding>
or <answer>...</answer>), nothing else; the caption inside <think> does not state image coordinates. Returns a list of
violation codes per exported record. No model calls.
"""
import re

TURN = re.compile(r'^\s*<think>\n?(?P<cap>.*?)\n?</think>\s*(?P<act><grounding>\{.*?\}</grounding>|<answer>.*?</answer>)\s*$', re.S)
COORD = [
    re.compile(r'\b[xy][12]?\s*[=:≈~]?\s*~?\s*0?\.\d{1,3}\b'),            # x 0.52, y=0.25, x1 .3
    re.compile(r'~\s*[xy]\s*0?\.\d'),                                       # ~x 0.27
    re.compile(r'\(\s*0?\.\d+\s*,\s*0?\.\d+\s*(,\s*0?\.\d+\s*,\s*0?\.\d+\s*)?\)'),  # (0.3, 0.4[, ...])
    re.compile(r'\[\s*0?\.\d+\s*,\s*0?\.\d+\s*,\s*0?\.\d+\s*,\s*0?\.\d+\s*\]'),       # [0.1,0.2,0.3,0.4]
    re.compile(r'\bbbox', re.I),
    re.compile(r'\(\s*\d{2,4}\s*,\s+\d{2,4}\s*\)'),                          # (195, 95) pixel points; not 550,000
]
PX = re.compile(r'\b\d{1,4}\s*px\b')
HOLD = {'layout_not_single_think_plus_action', 'caption_coordinates'}


def check_turn(text: str) -> list[str]:
    m = TURN.match(text)
    if not m:
        return ['layout_not_single_think_plus_action']
    cap = m.group('cap')
    out = ['caption_coordinates'] if any(p.search(cap) for p in COORD) else []
    if PX.search(cap):
        out.append('caption_pixel_measure')  # soft: pixel distances, reported but not held
    return out


def check_record(rec: dict) -> list[str]:
    out = []
    for c in rec['conversations']:
        if c['from'] == 'gpt':
            out += check_turn(c['value'])
    return sorted(set(out))
