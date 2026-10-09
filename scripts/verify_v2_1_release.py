#!/usr/bin/env python3
"""Recompute the additive V2.1 ledger; no pixel judgment or training claim."""
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def jl(name):return [json.loads(x) for x in (ROOT/'data'/name).read_text().splitlines() if x.strip()]
def main():
    baseline=jl('v2_merged_keep_trajectories_20261008.jsonl')
    add=jl('v2_1_increment_keep_20261008.jsonl')
    table=jl('v2_1_topup_review_table_20261008.jsonl')
    st=json.loads((ROOT/'data/v2_1_summary_20261008.json').read_text())
    assert len(baseline)==3953 and len(table)==st['process_reviewed']==37
    assert collections.Counter(r['verdict'] for r in table)==st['quality']
    assert len(add)==st['increment_KEEP']
    index={r['audit_id']:r for r in table}
    assert len(index)==len(table) and {r['audit_id'] for r in add}=={r['audit_id'] for r in table if r['verdict']=='KEEP'}
    for r in add:
        assert all(r[k]==v for k,v in index[r['audit_id']].items())
        assert r['verdict']=='KEEP' and r['disposition']=='core_ready'
        rv=r['review'];assert rv['answer_assessment']==rv['binding_assessment']=='supported'
        assert rv['recommendation']=='retain'
        assert sorted(rv['viewed_ids'])==sorted(v['view_id'] for v in r['views'])
        assert len(r['views'])==r['crops']+1==r['steps']==len(r['assistant_turns'])
        assert len(rv['crop_assessment'])==r['crops']
        assert {c['after_step'] for c in rv['crop_assessment']}==set(range(1,r['steps']))
        assert {c['returned_view'] for c in rv['crop_assessment']}=={v['view_id'] for v in r['views'][1:]}
        assert all(c['gain'] in {'new_detail','binding_context','comparison_evidence','informative_negative','honest_miss'} for c in rv['crop_assessment'])
        available=['original_image']
        for i,s in enumerate(r['step_context']):
            assert s['step']==i+1 and s['available_before']==available
            if i<r['crops']:
                assert r['views'][i+1]['parent'] in available
                available.append(r['views'][i+1]['view_id'])
        for v in r['views']:
            assert not v['asset_included'] and len(v['sha256'])==64
            assert not Path(v['source_path']).is_absolute() and '..' not in Path(v['source_path']).parts
        assert r['acceptance']=='model_reviewed_not_human_gold'
    allrows=baseline+add
    keys={r['canonical_sample_key'] for r in allrows};groups={g for r in allrows for g in r['canonical_groups']}
    assert len(keys)==len(allrows)
    assert not ({g for r in baseline for g in r['canonical_groups']} & {g for r in add for g in r['canonical_groups']})
    assert st['final_T_Q_G']==dict(T=len(allrows),Q=len(keys),G=len(groups))
    assert st['needed_for_over_4000']==max(0,4001-len(allrows))
    for model,n in st['by_model'].items():
        rs=[r for r in allrows if r['model']==model]
        assert len(rs)==n['total']
        assert sum(r['crops']==0 for r in rs)==n.get('direct',0)
        assert sum(r['crops']==1 for r in rs)==n.get('one_crop',0)
        assert sum(r['crops']>=2 for r in rs)==n.get('two_plus_crops',0)
    assert st['new_CPU']['exact_mask_passed']==37 and not st['human_acceptance'] and not st['training']
    print(json.dumps({'status':'PASS','increment':len(add),'T_Q_G':st['final_T_Q_G'],'remaining_to_over_4000':st['needed_for_over_4000']},indent=2))

if __name__=='__main__':main()
