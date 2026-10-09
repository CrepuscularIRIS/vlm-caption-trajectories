#!/usr/bin/env python3
"""Verify catalog completeness, provenance, step counts and publication boundary.

This verifies recorded review and export facts, not fresh visual correctness.
"""
import collections
import csv
import json
import re
from pathlib import Path
from build_trajectory_catalog import ROOT, SOURCES, digest, aggregate, clean_review


def load(path):
    return json.loads(path.read_text())


def main():
    web = ROOT / 'web'
    summary = load(web / 'data/summary.json')
    assert summary == load(ROOT / 'data/v2_2_complete_statistics_20261009.json')
    final, reviewed = {}, {}
    for phase, trace_name, review_name in SOURCES:
        for name, target in [(trace_name, final), (review_name, reviewed)]:
            path = ROOT / 'data' / name
            assert digest(path) == summary['source_inputs'][name]
            for line in path.read_text().splitlines():
                row = json.loads(line)
                assert row['audit_id'] not in target
                target[row['audit_id']] = row
    assert len(final) == 4040 and len(reviewed) == 5503
    assert len({r['canonical_sample_key'] for r in final.values()}) == summary['Q'] == 4040
    assert len({g for r in final.values() for g in r['canonical_groups']}) == summary['G'] == 3690
    assert len({r['canonical_sample_key'] for r in reviewed.values()}) == summary['reviewed_unique_tasks'] == 5378
    assert dict(collections.Counter(r['verdict'] for r in reviewed.values())) == summary['quality']
    assert summary['quality'] == {'KEEP': 4047, 'WEAK': 1120, 'BAD': 34, 'UNRESOLVED': 302}
    assert aggregate(list(final.values())) == summary['lengths']
    for model, counts in summary['by_model'].items():
        assert aggregate([r for r in final.values() if r['model'] == model]) == counts
    assert {str(k): v for k, v in collections.Counter(r['steps'] for r in final.values()).items()} == summary['step_histogram']
    index = {r['audit_id']: r for r in load(web / 'data/catalog.json')}
    assert index.keys() == reviewed.keys()
    shards = sorted((web / 'data').glob('paths-*.json'))
    details = {}
    forbidden_keys = {'question', 'source_path', 'source_result', 'result_path', 'image_path', 'reasoning_content', 'api_key'}
    def check_public(value):
        if isinstance(value, dict):
            assert not (forbidden_keys & value.keys()), forbidden_keys & value.keys()
            for v in value.values():
                check_public(v)
        elif isinstance(value, list):
            for v in value:
                check_public(v)
        elif isinstance(value, str):
            assert '<think>' not in value and '</think>' not in value
            assert not re.search(r'/(home|data|tmp)/', value)
    for shard in shards:
        rows = load(shard)
        check_public(rows)
        for d in rows:
            rid = d['audit_id']
            assert rid not in details
            details[rid] = d
            assert index[rid]['shard'] == str(shard.relative_to(web))
            assert d['review'] == clean_review(reviewed[rid]['review'])
            assert d['selected'] == (rid in final)
            assert not d['public_images']
            if rid not in final:
                assert 'assistant_turns' not in d
                continue
            f = final[rid]
            assert d['verdict'] == 'KEEP'
            assert d['assistant_turns'] == [t.replace('<think>', '<caption>').replace('</think>', '</caption>') for t in f['assistant_turns']]
            assert d['step_context'] == f['step_context']
            assert d['steps'] == d['crops'] + 1 == len(d['assistant_turns']) == len(d['views'])
            assert len(d['review']['crop_assessment']) == d['crops']
            assert d['review']['answer_assessment'] == d['review']['binding_assessment'] == 'supported'
            assert {c['gain'] for c in d['review']['crop_assessment']} <= {'new_detail', 'binding_context', 'comparison_evidence', 'informative_negative', 'honest_miss'}
            available = ['original_image']
            for i, step in enumerate(d['step_context']):
                assert step['step'] == i + 1 and step['available_before'] == available
                if i < d['crops']:
                    view = d['views'][i + 1]
                    assert view['parent'] in available
                    available.append(view['view_id'])
    assert details.keys() == reviewed.keys()
    with (ROOT / 'data/v2_2_complete_inventory_20261009.csv').open(encoding='utf-8-sig') as fp:
        csv_rows = list(csv.DictReader(fp))
    assert len(csv_rows) == 5503 and sum(r['selected'] == 'True' for r in csv_rows) == 4040
    manifest = load(web / 'catalog_manifest.json')
    assert manifest == {str(p.relative_to(web)): digest(p) for p in sorted((web / 'data').glob('*.json'))}
    assert not list(web.rglob('*.jpg')) and not list(web.rglob('*.png'))
    assert not summary['human_gold'] and not summary['training_run']
    print(json.dumps({'status': 'PASS', 'T': 4040, 'Q': 4040, 'G': 3690, 'reviewed': 5503, 'steps': 7047, 'crops': 3007, 'shards': len(shards), 'all_complete_visible_paths_matched': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
