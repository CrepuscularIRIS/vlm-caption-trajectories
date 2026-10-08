#!/usr/bin/env python3
"""Verify the public V2 ledger; this does not judge pixels or training quality."""
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'data'/name).read_text())
def jl(name):return [json.loads(s) for s in (ROOT/'data'/name).read_text().splitlines()]
def main():
    stats=read('v2_merged_summary_20261008.json')
    table=jl('v2_merged_review_table_20261008.jsonl')
    traces=jl('v2_merged_keep_trajectories_20261008.jsonl')
    assert stats['status']=='complete' and stats['pending_paths']==0
    assert stats['cpu_mask_check']['status']=='PASS'
    assert stats['cpu_mask_check']['exact_mask_and_target_passed']==5378
    assert len(table)==5378==stats['candidate_T_Q_G']['T']
    assert len({r['canonical_sample_key'] for r in table})==len(table)
    assert collections.Counter(r['verdict'] for r in table)=={g:n for g,n in stats['all']['quality'].items() if n}
    core=[r for r in table if r['disposition']=='core_ready']
    indexed={r['audit_id']:r for r in table}
    assert len(indexed)==len(table) and len({r['audit_id'] for r in traces})==len(traces)
    assert {r['audit_id'] for r in core}=={r['audit_id'] for r in traces}
    assert len(core)==len(traces)==stats['core_ready_T_Q_G']['T']==stats['core_ready_T_Q_G']['Q']
    assert len({g for r in traces for g in r['canonical_groups']})==stats['core_ready_T_Q_G']['G']
    assert len({r['canonical_sample_key'] for r in traces})==len(traces)
    for r in table:
        assert r['verdict'] in {'KEEP','WEAK','BAD','UNRESOLVED'}
        assert r['acceptance']=='model_reviewed_not_human_gold'
        assert len(r['result_sha256'])==64 and len(r['review_record_sha256'])==64
        assert r['steps']==r['crops']+1
        assert not Path(r['source_result']).is_absolute() and '..' not in Path(r['source_result']).parts
    for r in traces:
        assert all(r[k]==v for k,v in indexed[r['audit_id']].items()),'Ledger/trajectory mismatch'
        assert r['verdict']=='KEEP' and r['disposition']=='core_ready' and r['lf_passed']
        assert not r['quality_flags'] and not r['technical_errors'] and not r['review_format_errors']
        assert r['review']['recommendation']=='retain'
        assert r['review']['answer_assessment']==r['review']['binding_assessment']=='supported'
        crops=r['review']['crop_assessment']
        assert len(crops)==r['crops']
        assert {c['returned_view'] for c in crops}=={v['view_id'] for v in r['views'] if v['view_id']!='original_image'}
        assert all(c['gain'] in {'new_detail','binding_context','comparison_evidence','informative_negative','honest_miss'} for c in crops)
        assert {c['after_step'] for c in crops}==set(range(1,r['steps']))
        assert len(r['assistant_turns'])==r['steps'] and len(r['views'])==r['crops']+1
        assert len({v['view_id'] for v in r['views']})==len(r['views'])
        assert r['views'][0]['view_id']=='original_image' and r['views'][0]['parent'] is None
        available=['original_image']
        assert len(r['step_context'])==r['steps']
        for i,step in enumerate(r['step_context'],1):
            assert step['step']==i and step['available_before']==available
            if i<=r['crops']:
                new=r['views'][i]
                assert new['parent'] in available and new['view_id'] not in available
                assert next(c['returned_view'] for c in crops if c['after_step']==i)==new['view_id']
                available.append(new['view_id'])
        assert set(r['review']['viewed_ids'])=={v['view_id'] for v in r['views']}
        assert all(not v['asset_included'] and len(v['sha256'])==64 for v in r['views'])
        assert all(not Path(v['source_path']).is_absolute() and '..' not in Path(v['source_path']).parts for v in r['views'])
        assert all('<think>' in x and ('<grounding>' in x or '<answer>' in x) for x in r['assistant_turns'])
        assert '<answer>' in r['assistant_turns'][-1]
    for m,x in stats['by_model'].items():
        rs=[r for r in table if r['model']==m];accepted=[r for r in rs if r['disposition']=='core_ready']
        assert len(rs)==x['paths'] and len(accepted)==x['core_ready']
        assert sum(r['crops']==0 for r in accepted)==x['core_direct']
        assert sum(r['crops']==1 for r in accepted)==x['core_one_crop']
        assert sum(r['crops']>=2 for r in accepted)==x['core_two_plus_crops']
    print(json.dumps({'integrity':'PASS','candidate_paths':len(table),'core_T_Q_G':stats['core_ready_T_Q_G'],
                     'human_gold':False,'training_gain_claimed':False},indent=2))

if __name__=='__main__':main()
