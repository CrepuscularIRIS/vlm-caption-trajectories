"""Construction-time fixes and the online step monitor ("drag") for the Mini-o3 harness.

- lenient_parse: recover {thought, action} when the generator breaks the JSON (unterminated
  action string, Table-6 "Thought:/Action:" text format, stray LaTeX escapes).
- expand_min_crop: crops smaller than a minimum source-pixel size are enlarged around their
  centre (zooming past native resolution adds no information and invited fabricated detail).
- StepMonitor: a different-family model checks one step (claims vs pixels, action sanity)
  WITHOUT the gold answer; on "revise" the generator redoes that step with a location-only hint.
"""
import json
import re

from probe_api import structured

ACTION = re.compile(r'(grounding\(\s*\[[^\]]*\][^)]*\)|final\((?:[^()]|\([^()]*\))*\))', re.S)
THOUGHT_JSON = re.compile(r'"thought"\s*:\s*"(.*?)"\s*,\s*"action"', re.S)
THOUGHT_TEXT = re.compile(r'Thought\s*\d*\s*:\s*(.*?)\n\s*Action\s*\d*\s*:', re.S)


def lenient_parse(content: str) -> dict | None:
    """Return {'thought','action'} from malformed output, or None."""
    hits = ACTION.findall(content)
    if not hits:
        return None
    action = hits[-1].replace('\\"', '"').strip()
    m = THOUGHT_JSON.search(content) or THOUGHT_TEXT.search(content)
    thought = m.group(1) if m else content[:content.rfind(hits[-1])]
    thought = thought.replace('\\"', '"').replace('\\n', '\n').strip().strip('`{').strip()
    return {'thought': thought, 'action': action}


def expand_min_crop(ab, W, H, min_px):
    """Enlarge an absolute box to at least min_px on each side, clamped to the image."""
    x0, y0, x1, y1 = ab
    grew = False
    for lo, hi, size in ((0, 2, W), (1, 3, H)):
        box = [x0, y0, x1, y1]
        if box[hi] - box[lo] < min_px:
            c = (box[lo] + box[hi]) / 2
            a = max(0.0, min(c - min_px / 2, size - min_px))
            box[lo], box[hi] = a, min(size, a + min_px)
            x0, y0, x1, y1 = box
            grew = True
    return (x0, y0, x1, y1), grew


MONITOR_TASK = """TASK: you review ONE step of another model's visual search. Do NOT answer the question yourself.
You do not know the correct answer. Image 1 is the original image. Image 2 (if present) is the most recent
zoomed observation, i.e. what the step is looking at now.
Request a revision ONLY for a clear problem:
- a visual claim in the thought that is not visible or is contradicted by the pixels;
- a final answer although the evidence for it is not clearly visible;
- zooming into a region that cannot contain the target;
- a zoom so small that the result is only blurry pixels;
- ignoring evidence that is already clearly visible.
Merely suboptimal steps (could crop tighter, could be more efficient) must PASS.
The hint may only NAME the problem (e.g. "the claim that the letters are readable is not supported by the crop",
"the crop is too small to add detail", "this region does not seem to contain the target; reconsider where it is").
The hint must NOT contain coordinates, numbers, colors, text you read, or any attribute of the target.
Return ONLY JSON: {"verdict":"pass|revise","problems":["..."],"hint":"..."}"""

NUMBERS = re.compile(r'\d+(?:\.\d+)?')


def sanitize_hint(text: str) -> str:
    """Belt-and-braces: strip numbers/coordinates the monitor was told not to give."""
    return NUMBERS.sub('#', text)


def leak_flag(hint: str, gold: str) -> bool:
    """Offline audit only (gold never reaches any model): does a hint contain a gold token?"""
    toks = [t for t in re.findall(r'[a-z0-9]+', gold.lower()) if len(t) > 2]
    return any(re.search(r'\b' + re.escape(t) + r'\b', hint.lower()) for t in toks)


REVISE_NOTE = ("A reviewer checked your previous step and asked you to redo it. Hint: {hint}\n"
               "Redo step {i}. Write the thought as your own reasoning from the images; do not mention the reviewer.")


class StepMonitor:
    def __init__(self, api, max_tokens=3000):
        self.api, self.max_tokens = api, max_tokens

    def review(self, doc_id, question, original, latest, i, attempt, thought, action) -> dict:
        content = [{'type': 'text', 'text': MONITOR_TASK + f'\n\nQuestion: {question}\nImage 1 (original):'},
                   self.api.image(str(original))]
        if latest is not None:
            content += [{'type': 'text', 'text': 'Image 2 (latest observation):'}, self.api.image(str(latest))]
        content.append({'type': 'text', 'text': f'Step {i}\nThought: {thought}\nAction: {action}\n\n'
                                                'Now output ONLY the JSON review.'})
        r = self.api.call(f'{doc_id}/monitor_t{i}_a{attempt}', [{'role': 'user', 'content': content}],
                          self.max_tokens, replicate_id='monitor', temperature=0.)
        obj = structured(r['content'])
        if obj is None:
            m = re.search(r'\{.*\}', r['content'], re.S)
            obj = structured(m.group(0)) if m else None
        verdict = (obj or {}).get('verdict')
        return dict(attempt=attempt, request_sha256=r['request_sha256'], finish_reason=r['finish_reason'],
                    verdict=verdict if verdict in ('pass', 'revise') else 'unparsed',
                    problems=(obj or {}).get('problems') or [], hint=str((obj or {}).get('hint') or ''))


def revise_message(i: int, review: dict) -> str:
    # Only the constrained hint goes back; the monitor's `problems` text stays in the ledger
    # (A/B 2026-09-30: problems text leaked perceptual content twice).
    return REVISE_NOTE.format(hint=sanitize_hint(review['hint']) or 'look again carefully', i=i)


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
