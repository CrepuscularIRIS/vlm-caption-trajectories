"""Validate published audit structure/provenance, not scientific annotations."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selected', type=Path, help='Optional private source selected.json')
    args = ap.parse_args()
    base = Path(__file__).resolve().parent
    audit = json.loads((base / 'step_annotations.json').read_text())
    questions = json.loads((base / 'questions.json').read_text())
    rows = audit['rows']
    assert len(rows) == 49
    assert len({r['case_id'] for r in rows}) == 10
    assert {r['uid'] for r in rows} == {q['uid'] for q in questions}
    assert len(questions) == 8
    assert sum(r['returned_view'] is not None for r in rows) == 39
    image_hashes = set()
    view_positions = {}
    seen = set()
    for r in rows:
        key = (r['case_id'], r['step'])
        assert key not in seen
        seen.add(key)
        assert r['source_round'] == r['step'] - 1
        assert [v['view'] for v in r['available_views']] == list(range(r['step']))
        assert r['causal_crop_utility'] == 'not_identified'
        assert r['focal_claim_status'] in audit['status_definition']
        vs = r['available_views'] + ([r['returned_view']] if r['returned_view'] else [])
        if r['returned_view']:
            assert r['requested_action']['kind'] == 'grounding'
            assert r['returned_view']['view'] == r['step']
            assert r['executed_box_abs'] == r['returned_view']['box_abs']
        for v in vs:
            actual = hashlib.sha256((base / v['image']).read_bytes()).hexdigest()
            assert actual == v['sha256']
            image_hashes.add(actual)
            pos = (r['case_id'], v['view'])
            assert pos not in view_positions or view_positions[pos] == actual
            view_positions[pos] = actual
    assert len(image_hashes) == 37 and len(view_positions) == 49
    for case, count in Counter(r['case_id'] for r in rows).items():
        assert {s for c, s in seen if c == case} == set(range(1, count + 1))
    for file in [base / 'README.md', base / 'TIMELINE.md']:
        for link in re.findall(r'\]\(([^)]+)\)', file.read_text()):
            if '://' not in link:
                assert (base / link.split('#')[0]).exists(), link
    if args.selected:
        assert hashlib.sha256(args.selected.read_bytes()).hexdigest() == audit['source_selected_sha256']
        source = json.loads(args.selected.read_text())
        for c in source:
            rs = [r for r in rows if r['case_id'] == c['id']]
            assert len(rs) == len(c['trajectory']['turns'])
            for r, t in zip(rs, c['trajectory']['turns']):
                assert r['generated_text'] == t['text']
                assert r['requested_action'] == t['parsed']
                assert r['executed_box_abs'] == t.get('box_abs')
                assert r['text_in_history'] == t.get('text_in_history', t['text'])
    print('PASS: 10 trajectories, 49 turns, 49 view positions, 37 image hashes, 8 full questions; source match=' + str(bool(args.selected)))


if __name__ == '__main__':
    main()
