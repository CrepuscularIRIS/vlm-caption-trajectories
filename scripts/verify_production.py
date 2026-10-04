#!/usr/bin/env python3
"""Verify frozen public metadata, case timelines, assets and safe relative paths.
This is not a visual judge and does not establish SFT acceptance or model gains.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return json.loads((ROOT / rel).read_text())


def require(ok, why):
    if not ok:
        raise ValueError(why)


def relative(s):
    p = Path(s)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe published path: ' + s)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    s = read('data/production_20261003.json')
    rows = [json.loads(x) for x in (ROOT / 'data/production_audit_rows_20261003.jsonl').read_text().splitlines()]
    require(len(rows) == s['audit']['n'] == 2520, 'Audit row count')
    require(len({r['key'] for r in rows}) == len(rows), 'Repeated audit key')
    require(dict(Counter(r['pool'] for r in rows)) == s['audit']['pools'], 'Pool distribution')
    for tag, counts in s['audit']['by_tag'].items():
        require(dict(Counter(r['pool'] for r in rows if r['tag'] == tag)) == counts, 'Pool by tag')
    clean = [r for r in rows if r['pool'] == 'clean_candidate']
    c = s['audit']['clean']
    require(len(clean) == c['n'] == 1241, 'Clean candidate count')
    require(len({r['group_id'] for r in clean}) == c['groups'] == 1128, 'Independent groups')
    require(Counter(str(r['steps']) for r in clean) == c['step_histogram'], 'Step histogram')
    require(Counter(r['material'] for r in clean) == c['material'], 'Material histogram')
    for field in ('fallback_unreviewed','fallback_true','fallback_false','fallback_labels','holds','revises','fallback_candidate_events'):
        require(sum(r[field] for r in clean) == c[field], 'Behavior total: ' + field)
    require(sum(r['fallback_true'] > 0 for r in clean) == c['fallback_true_traces'], 'Fallback trace total')
    require(sum(r['crops'] > 0 for r in clean) == c['crops'] == 722, 'Crop candidates')
    require(sum(r['crops'] == 0 for r in clean) == c['direct'] == 519, 'Direct candidates')
    require(Counter(r['reader'] for r in clean) == {'gemini-3.8-flash-low':235,'grok-4.7':1006}, 'Reader distribution')
    require([r['key'] for r in clean if r['fallback_unreviewed']] == read('data/fallback_pending_20261003.json'), 'Fallback queue')
    require(all(r['decision'] == 'AUTO_PASS_PENDING_MANUAL' for r in clean), 'Candidate != human acceptance')
    for r in rows:
        for k in ('result_path','audit_path'):
            relative(r[k])
        for k in ('result_sha256','audit_sha256'):
            require(len(r[k]) == 64, 'Source digest length')
    require(sum(m['n'] for m in s['models'].values()) == s['generation_total'] == 32333, 'Generation aggregate')
    require(sum(s['funnel']['difficulty'].values()) == s['generation_items'], 'Difficulty count')
    require(sum(s['funnel']['route'].values()) == s['generation_items'], 'Route count')
    for m in s['models'].values():
        require(m['cpu_crop'] + m['cpu_direct'] == m['cpu_candidates'], 'CPU decomposition')
    prov = read('data/production_provenance_20261004.json')
    assets = read('media/20261004/ATTRIBUTION.json')
    used, ntraces = set(), 0
    case_paths = sorted((ROOT / 'cases/20261004').glob('*.json'))
    require(len(case_paths) == 9, 'Case count')
    for cp in case_paths:
        case = json.loads(cp.read_text())
        ts = case['trajectories']
        require(len({t['family'] for t in ts}) == case['independent_generator_families'], 'Family count')
        for t in ts:
            ntraces += 1
            relative(t['source_path'])
            require(prov['source_files'][t['source_path']] == t['source_sha256'], 'Case provenance')
            require(t['manual_verdict'] is None, 'Do not invent manual acceptance')
            views = {v['view_id']:v for v in t['views']}
            require(len(views) == len(t['views']) == t['crops'] + 1, 'Unique views')
            available = ['original_image']
            for i, step in enumerate(t['steps'], 1):
                require(step['step'] == i and step['available_before'] == available, 'Prefix timeline')
                require(set(step['cited']) <= set(available), 'Future cited view')
                if step['observation']:
                    v = views[step['observation']]
                    require(v['view_id'] not in available and v['parent'] in available, 'Invalid crop parent')
                    require(step['action']['source'] == v['parent'], 'Action/source mismatch')
                    available.append(v['view_id'])
            require(set(available) == set(views), 'Unreached view')
            for v in views.values():
                relative(v['source_path'])
                if v['public_asset']:
                    relative(v['public_asset'])
                    require(case['item_id'].startswith('worldbench_'), 'Only allowlisted WorldBench images may be public')
                    require(digest(ROOT / v['public_asset']) == v['sha256'], 'Image bytes changed')
                    require(assets[v['public_asset']]['sha256'] == v['sha256'], 'Image attribution hash')
                    require(assets[v['public_asset']]['license'] == 'CC-BY-4.0', 'Asset license')
                    used.add(v['public_asset'])
            if case['case_id'] in ('C07','C08','C09'):
                require(t['score']['score'] == 0, 'Shared gold disagreement score')
    require(ntraces == 18 and used == set(assets) and len(used) == 8, 'Curated trajectory/asset count')
    for rel, meta in prov['published_code'].items():
        relative(rel)
        if meta['transformation'] == 'byte-for-byte reading snapshot':
            require(digest(ROOT / rel) == meta['source_sha256'], 'Reference code bytes changed')
    print(json.dumps(dict(consistency='PASS',audit_rows=len(rows),automatic_candidates=len(clean),
        candidate_groups=c['groups'],fallback_pending=81,cases=len(case_paths),trajectories=ntraces,public_views=len(used)),indent=2))
    print('PASS is metadata consistency. Human acceptance and production export remain pending.')


if __name__ == '__main__':
    main()
