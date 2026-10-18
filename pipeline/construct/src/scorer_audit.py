"""Scorer audit over every stored result (2026-10-07; NEXT_STEPS P0). CPU only, no score is overwritten.

For each path that ended with an answer and is not stored as correct, flags why the stored 0 / None may be a measurement
problem rather than a wrong answer. Writes ANOMALIES.jsonl (one row per flagged path) and SUMMARY.json.
"""
import collections
import json
import logging
import pickle
import re
from pathlib import Path

from PIL import Image

from rescore_open import _iou, open_ok

logger = logging.getLogger(__name__)
OUT = Path('${ICML_DIR}/obs/runs/scorer_audit_20261007')
OPT = re.compile(r'^\s*\(?([A-Z])[\).．:：]\s*(.+?)\s*$')


def norm(s) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def options(q: str) -> dict[str, str]:
    out = {}
    for line in q.splitlines():
        m = OPT.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    if len(out) < 2:  # inline "A．x, B．y"
        for m in re.finditer(r'([A-H])[．.\)]\s*([^,;]+?)(?=[,;]\s*[A-H][．.\)]|$)', q):
            out.setdefault(m.group(1), m.group(2).strip())
    return out


def num(s):
    m = re.fullmatch(r'\s*\$?(-?[\d,]*\.?\d+)\s*%?\s*', str(s))
    return float(m.group(1).replace(',', '')) if m else None


def boxes(a):
    if isinstance(a, str):
        v = [float(x) for x in re.findall(r'-?\d+\.?\d*', a)]
        a = [v[i:i + 4] for i in range(0, len(v) - len(v) % 4, 4)]
    elif a and not isinstance(a[0], list):
        a = [a]
    return a


def match(a, g) -> bool:
    if len(a) != len(g):
        return False
    used = set()
    for gb in g:
        best = max(((_iou(ab, gb), j) for j, ab in enumerate(a) if j not in used), default=(0.0, None))
        if best[0] < .5:
            return False
        used.add(best[1])
    return True


def flag(r: dict) -> str | None:
    a, g, kind = r['answer'], r['gold'], r['kind']
    if a is None:
        return None
    if kind == 'letter':
        gs, as_ = str(g).strip(), str(a).strip()
        opts = options(r['question'])
        if not re.fullmatch(r'[A-Za-z]', gs):
            return 'letter_gold_not_a_letter:' + ('answer_equals_gold' if norm(as_) == norm(gs) else 'answer_differs')
        if as_.upper() == gs.upper():
            return 'letter_equal_but_zero'
        m = re.fullmatch(r'\(?([A-Za-z])[\).:]?(\s.*)?', as_)
        if m and m.group(1).upper() == gs.upper() and len(as_) > 1:
            return 'letter_with_extra_text_matches_gold'
        if not re.fullmatch(r'[A-Za-z]', as_) and opts.get(gs.upper()) and norm(opts[gs.upper()]) == norm(as_):
            return 'answer_is_gold_option_text'
        if not re.fullmatch(r'[A-Za-z]', as_):
            return 'letter_answer_not_a_letter_other'
        if as_.upper() not in opts and len(opts) >= 2:
            return 'answer_letter_outside_listed_options'
        return None
    if kind == 'short':
        if norm(a) == norm(g):
            return 'short_normalized_equal_but_zero'
        na, ng = num(a), num(g)
        if na is not None and ng is not None and abs(na - ng) < 1e-9:
            return 'short_numeric_equal_but_zero'
        if ng is not None and re.search(rf'(?<![\d.]){re.escape(str(g).strip())}(?![\d.])', str(a)):
            return 'short_answer_contains_gold_number'
        if norm(g) and (norm(g) in norm(a) or (len(norm(a)) >= 3 and norm(a) in norm(g))):
            return 'short_text_containment'
        if r['method'] == 'number_exact' and ng is None:
            return 'short_number_method_on_text_gold'
        return None
    if kind == 'open':
        if open_ok(a, g):
            return 'open_set_equal_unscored'
        sa = {norm(x) for x in re.split(r'[,;\n]', str(a)) if norm(x)}
        sg = {norm(x) for x in re.split(r'[,;\n]', str(g)) if norm(x)}
        if len(sg) == 1 and next(iter(sg)) in norm(a):
            return 'open_answer_contains_gold'
        if sa and sg and (sa < sg or sg < sa):
            return 'open_subset_or_superset'
        return 'open_unscored_differs'
    if kind == 'bbox':
        try:
            ab, gb = boxes(a), boxes(g)
        except (ValueError, TypeError):
            return 'bbox_unparseable'
        if not ab:
            return 'bbox_unparseable'
        if match(ab, gb):
            return 'bbox_pixel_match_unscored'
        if all(0 <= v <= 1.0 for b in ab for v in b):
            try:
                w, h = Image.open(r['image']).size
            except OSError:
                return 'bbox_normalized_answer_image_missing'
            return 'bbox_normalized_matches_after_scaling' if match([[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in ab], gb) \
                else 'bbox_normalized_no_match'
        return 'bbox_pixel_no_match'
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    rows = pickle.load((OUT / 'ALL_ROWS.pkl').open('rb'))
    todo = [r for r in rows if r['term'] == 'answer' and r['score'] != 1]
    out, by = [], collections.Counter()
    for r in todo:
        f = flag(r)
        by[(r['bench'], r['kind'], r['method'], f)] += 1
        if f:
            out.append(dict(item_id=r['item_id'], bench=r['bench'], model=r['model'], kind=r['kind'], method=r['method'], stored_score=r['score'],
                            gold=r['gold'], answer=r['answer'], flag=f, result_path=r['path']))
    with (OUT / 'ANOMALIES.jsonl').open('w') as fh:
        for o in out:
            fh.write(json.dumps(o, ensure_ascii=False) + '\n')
    flags = collections.Counter(o['flag'] for o in out)
    tasks = {f: len({o['item_id'] for o in out if o['flag'] == f}) for f in flags}
    model = collections.Counter((o['flag'], o['model']) for o in out)
    (OUT / 'SUMMARY.json').write_text(json.dumps(dict(
        all_results=len(rows), answered_not_stored_correct=len(todo), flagged_paths=len(out), by_flag_paths=dict(flags), by_flag_tasks=tasks,
        by_flag_model={f'{k[0]}|{k[1]}': v for k, v in sorted(model.items())},
        by_bench_kind_method_flag={'|'.join(map(str, k)): v for k, v in sorted(by.items(), key=str)}), indent=1, ensure_ascii=False))
    for f, n in flags.most_common():
        logger.info('%-45s paths=%d tasks=%d', f, n, tasks[f])


if __name__ == '__main__':
    main()
