"""v4c: which episodes count as correct, and which tasks a later generator still has to do. Offline only.

The frozen scorer (``corpus_scoring``) cannot accept four kinds of correct answers, found in the 2026-10-13
audit of the 224 tasks v4 left without a candidate (reports/interp_v4_scoring_audit_20261013):
  * multiple choice whose option labels are not A-J (VisReason: "1".."5", "Z");
  * open answers (no rule at all), accepted here only when equal to the reference after normalization;
  * box answers (no rule at all), accepted here when every reference box is matched with IoU >= 0.5;
  * short text answers that contain the reference words plus an article or modifier.
Everything else is the frozen rule. An episode must still have ended in an answer, passed the protocol and
carry correct supervision flags. GT is read here and nowhere near a generator.
"""
import ast
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

IOU_MIN = 0.5
FROZEN, LABEL, OPEN, BOX, SHORT = 'frozen_scorer', 'option_label_not_A_to_J', 'open_normalized_equal', 'bbox_iou', 'short_gold_words_contained'


def _norm(text: object) -> str:
    return re.sub(r'[^\w]', '', str(text or '').lower())


def _words(text: object) -> List[str]:
    return re.findall(r'\w+', str(text or '').lower())


def _iou(a: List[float], b: List[float]) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _boxes_match(answer: str, gold: List[List[float]], source_px: List[int]) -> bool:
    try:
        boxes = ast.literal_eval(answer.strip())
    except (ValueError, SyntaxError):
        return False
    if not isinstance(boxes, (list, tuple)) or not boxes:
        return False
    if isinstance(boxes[0], (int, float)):
        boxes = [boxes]
    if len(boxes) != len(gold) or any(not isinstance(b, (list, tuple)) or len(b) != 4 for b in boxes):
        return False
    if max(max(b) for b in boxes) <= 1.5:  # normalized to the original image
        width, height = source_px
        boxes = [[b[0] * width, b[1] * height, b[2] * width, b[3] * height] for b in boxes]
    return all(max(_iou(g, b) for b in boxes) >= IOU_MIN for g in gold)


def accepted(result: dict, spec: dict) -> Optional[str]:
    """Name of the rule under which this episode is a correct trajectory, or None."""
    checks, ro = result['objective_checks'], result['rollout']
    if checks.get('eligible_without_process_review'):
        return FROZEN
    if ro['termination'] != 'answer' or not checks.get('protocol_passed') or not checks.get('editor_supervision_flags_correct'):
        return None
    answer, kind, gold = ro.get('answer') or '', spec['kind'], spec.get('gold')
    if kind == 'letter' and not re.fullmatch(r'[A-Ja-j]', str(gold).strip()):
        return LABEL if _norm(answer) == _norm(gold) else None
    if kind == 'open':
        return OPEN if _norm(answer) == _norm(gold) else None
    if kind == 'bbox':
        return BOX if _boxes_match(answer, gold, ro['views'][0]['source_px']) else None
    if kind == 'short' and not re.search(r'\d', str(gold)):
        have = _words(answer)
        return SHORT if _words(gold) and all(w in have for w in _words(gold)) else None
    return None


def episodes(parent: Path) -> Iterable[dict]:
    """Saved results of every wave of a campaign, with the rule that accepts each (or None).

    A campaign directory that was never built (an optional later lane) has no episodes.
    """
    if not parent.exists():
        return
    specs ={p['audit_id']: p['answer_spec'] for p in json.loads((parent/'SELECTION_PRIVATE.json').read_text())}
    for path in sorted(parent.glob('continuations/*/sol/sol/results/*.json')):
        result = json.loads(path.read_text())
        yield dict(audit_id=path.stem, path=path, result=result, rule=accepted(result, specs[path.stem]))


def issued_ids(parents: Iterable[Path]) -> Set[str]:
    return {e['audit_id'] for parent in parents for e in episodes(parent)}


def accepted_ids(parents: Iterable[Path]) -> Dict[str, str]:
    return {e['audit_id']: e['rule'] for parent in parents for e in episodes(parent) if e['rule']}


def write_exclusion(scratch: Path, name: str, ids: Set[str]) -> Path:
    """A scope file the frozen v4 build reads as 'already issued elsewhere'."""
    target = scratch/name
    target.mkdir(parents=True)
    (target/'SCOPE_ALL_4040.json').write_text(json.dumps([dict(audit_id=a) for a in sorted(ids)], indent=1))
    return target


def build_fill(v4, target: Path, solved_by: List[Path], own: List[Path], variant_file: Path) -> dict:
    """Frozen v4 build over the tasks no earlier generator solved and this model has not run. Sends nothing."""
    for parent in solved_by:
        for state in parent.glob('continuations/*/sol/RUN_STATE.json'):
            if json.loads(state.read_text())['state'] == 'running':
                raise ValueError('Earlier generator still running: ' + str(state.parent))
    solved = set(accepted_ids(solved_by))
    tried = issued_ids(own) - solved
    scratch = target.parent/(target.name + '_exclusions')
    excluded = [write_exclusion(scratch, 'solved_by_earlier_generator', solved),
                write_exclusion(scratch, 'already_run_by_this_model', tried)]
    variant = dict(v4.plan.DEFAULT_VARIANT, **json.loads(variant_file.read_text()))
    return dict(v4.build(target, v4.SOURCE, None, excluded, variant), solved_by_earlier_generator=len(solved),
                already_run_by_this_model=len(tried))
