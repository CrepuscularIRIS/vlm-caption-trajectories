#!/usr/bin/env python3
"""Offline consistency checks for the October 8 metadata release, not visual review."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    s = json.loads((ROOT/'data/sft_audit_20261008.json').read_text())
    c = s['candidate_selection']
    path = ROOT/c['manifest']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == c['manifest_sha256']
    rows = [json.loads(line) for line in path.open()]
    assert len(rows) == c['T_candidate'] == 5277
    assert len({r['canonical_sample_key'] for r in rows}) == c['Q'] == len(rows)
    assert len({r['source_result'] for r in rows}) == len(rows)
    assert Counter(r['grade'] for r in rows) == c['quality'] == {'KEEP': 5277}
    groups = Counter(g for r in rows for g in r['canonical_groups'])
    assert len(groups) == c['G'] == 4755
    assert {str(k): v for k, v in Counter(groups.values()).items()} == c['group_task_count_histogram']
    assert sum(n>1 for n in groups.values()) == c['repeated_image_groups'] == 458
    assert max(groups.values()) == c['max_tasks_per_group'] == 17
    for r in rows:
        assert len(r['canonical_groups']) == 1
        assert r['steps'] == r['crops']+1
        assert r['acceptance_status'] == 'model_reviewed_candidate_not_final_accepted_SFT'
        assert len(r['result_sha256']) == 64
        p = Path(r['source_result'])
        assert not p.is_absolute() and '..' not in p.parts
    for m, expected in c['by_model'].items():
        rs = [r for r in rows if r['model']==m]
        actual = dict(paths=len(rs), direct=sum(r['crops']==0 for r in rs),
                      one_crop=sum(r['crops']==1 for r in rs),
                      two_plus_crops=sum(r['crops']>=2 for r in rs))
        assert actual == expected, (m, actual, expected)
    assert Counter(r['bench'] for r in rows) == c['by_dataset']
    assert Counter(r['material'] for r in rows) == c['by_material']
    inv = s['inventory']
    quality = Counter()
    for d in inv['by_model'].values():
        assert sum(d['quality'].values()) == sum(d['termination'].values()) == d['paths']
        assert sum(d['crops'].values()) == d['paths']
        assert sum(d['keep_crops'].values()) == d['quality'].get('KEEP', 0)
        quality.update(d['quality'])
    assert quality == inv['quality']
    assert sum(quality.values()) == inv['total'] == inv['main']+inv['supplemental'] == 86474
    assert inv['records_with_any_review'] == inv['total']-quality['NOT_REVIEWED'] == 14725
    assert sum(quality[k] for k in ['KEEP','WEAK','BAD','UNRESOLVED']) == 10792
    assert all(v is None for v in s['accepted_sft'].values())
    un = s['unreviewed_screen']
    assert un['records'] == quality['NOT_REVIEWED'] == sum(un['termination'].values())
    assert un['records'] >= un['corrected_cpu_candidates'] >= un['corrected_cpu_tool_paths'] >= un['corrected_cpu_two_plus_crop_paths']
    print(json.dumps({'consistency': 'PASS', 'inventory': inv['total'],
                      'candidate_T_Q_G': [len(rows), c['Q'], len(groups)],
                      'duplicate_tasks': 0, 'accepted_sft': s['accepted_sft']}, indent=2))
    print('Metadata consistency is not visual truth or final SFT acceptance.')


if __name__ == '__main__':
    main()
