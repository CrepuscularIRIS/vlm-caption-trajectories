"""Deterministic answer scoring for corpus_v1 (no model judge).

answer_spec kinds:
  letter : multiple choice; gold is one letter; prediction must reduce to exactly one option letter.
  short  : short answer; numeric comparison when the gold carries a number, else normalized text match
           against gold and aliases.
  open   : open-ended; not stably checkable -> score None (calibration pool / model judge later).
  bbox   : localization; gold boxes in source pixels -> score None until a box answer format is frozen.
A score of None means "not scored", never "wrong".
"""
import re

LETTER_RE = re.compile(r'^\s*\(?([A-Ja-j])\)?\s*(?:[.:)\]]\s*.*)?$', re.S)
NUM_RE = re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?')
ARTICLES = re.compile(r'\b(a|an|the)\b')
WORD_NUMS = {w: str(i) for i, w in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '
    'seventeen eighteen nineteen twenty'.split())}


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = ARTICLES.sub(' ', s)
    return ' '.join(WORD_NUMS.get(w, w) for w in s.split())


def first_number(s: str) -> float | None:
    s = ' '.join(WORD_NUMS.get(w, w) for w in re.sub(r'[^\w\s.,-]', ' ', s.lower()).split())
    m = NUM_RE.search(s)
    return float(m.group().replace(',', '')) if m else None


def score_letter(gold: str, pred: str, n_options: int | None) -> int:
    m = LETTER_RE.match(pred or '')
    if not m:
        return 0
    letter = m.group(1).upper()
    if n_options and ord(letter) - 65 >= n_options:
        return 0
    return int(letter == gold.strip().upper())


def score_short(gold: str, pred: str, aliases: list[str] | None = None) -> int:
    g_num = first_number(gold)
    if g_num is not None:
        p_num = first_number(pred or '')
        return int(p_num is not None and abs(p_num - g_num) < 1e-9)
    targets = {norm_text(gold)} | {norm_text(a) for a in (aliases or [])}
    return int(norm_text(pred or '') in targets)


def score(spec: dict, pred: str) -> dict:
    kind = spec['kind']
    if kind == 'letter':
        return {'score': score_letter(spec['gold'], pred, spec.get('n_options')), 'method': 'letter_exact'}
    if kind == 'short':
        method = 'number_exact' if first_number(spec['gold']) is not None else 'text_normalized'
        return {'score': score_short(spec['gold'], pred, spec.get('aliases')), 'method': method}
    return {'score': None, 'method': f'unscored_{kind}'}
