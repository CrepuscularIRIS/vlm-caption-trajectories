"""Freeze a task-deduplicated V2 review queue from unchanged existing paths."""
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path('${ICML_DIR}');OUT=Path(__file__).resolve().parent
OLD=ROOT/'reports/sft_final_acceptance_20261008'
def jl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def savejl(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))

def main():
    for s in ('packets','records','queues','reused_originals'): (OUT/s).mkdir(exist_ok=True)
    baseline=jl(OLD/'MANIFEST.jsonl'); hist={x['audit_id']:x for x in jl(OLD/'HISTORY_ENRICHED.jsonl')}
    candidates=jl(ROOT/'reports/sft_v1_v2_reconciliation_20261008/V2_KEEP_CANDIDATES.jsonl')
    priority=[r for r in candidates if r.get('crops_informative')=='all_useful']
    bykey=collections.defaultdict(list)
    for r in priority:bykey[r['canonical_sample_key']].append(r)
    basekeys={r['canonical_sample_key'] for r in baseline}
    new=[];alternatives=[]
    # No crop-count reward: evidence and binding first, then lower redundancy,
    # action fit, imitatability, shorter adequate path, stable path hash.
    def rank(r):return (-int(r.get('evidence') or 0),-int(r.get('binding') or 0),
        r.get('redundancy')!='low',-int(r.get('action_fit') or 0),-int(r.get('imitate') or 0),
        r['crops'],hashlib.sha256(r['result_path'].encode()).hexdigest())
    for k,rs in sorted(bykey.items()):
        rs.sort(key=rank)
        if k in basekeys:alternatives.extend(dict(r,reason='same_task_V1_preserved') for r in rs)
        else:new.append(rs[0]);alternatives.extend(dict(r,reason='same_new_task_lower_rank') for r in rs[1:])
    rows=[];reuse=0;pending=[]
    for old in baseline:
        r=dict(old,source_version='V1',origin_audit_id=old['audit_id'],historical_reviews=hist[old['audit_id']]['reviews'])
        rows.append(r)
    for r in new:
        p=Path(r['result_path']);x=json.loads(p.read_text());item=x['item']
        rows.append(dict(r,source_version='V2_increment',origin_audit_id=None,
            result_sha256=sha(p),source_verified_sha256=sha(p),stage='main_four',material=item.get('material'),
            canonical_groups=[item['group_id']],source_split=item.get('split'),
            source_image_sha256=item.get('image_sha256'),gold_metadata=item.get('answer_spec'),
            technical_findings=[],review_source='CC_multistep_seg1_5_Sonnet55',
            historical_reviews=[dict(r,reviewer_attribution='Sonnet 5.5',
                attribution_evidence='RESUME_DONE batch declaration; receipts sampled, not all individually reverified')]))
    # Original F ids remain stable; added rows use I ids. Review increment first.
    for n,r in enumerate(rows):
        aid=r['origin_audit_id'] or f'I{n-len(baseline)+1:05d}';r['audit_id']=aid
        rp=Path(r['result_path']);assert sha(rp)==r['result_sha256']
        if r['origin_audit_id']:
            packet=json.loads((OLD/'packets'/f'{aid}.json').read_text())
        else:
            x=json.loads(rp.read_text());ro=x['rollout'];views=[]
            for v in ro['views']:
                vp=Path(v['path']);h=sha(vp);assert not v.get('sha256') or h==v['sha256']
                views.append({**{k:v.get(k) for k in ('view_id','parent','path','shown_px','fov_original')},'sha256':h})
            packet={'audit_id':aid,'item_id':r['item_id'],'question':x['item']['question'],'views':views,
                'steps':[{k:s.get(k) for k in ('step','available_before','parsed','observation','requested_box')} for s in ro['steps']],
                'answer':ro.get('answer'),'termination':ro.get('termination')}
        (OUT/'packets'/f'{aid}.json').write_text(json.dumps(packet,ensure_ascii=False,indent=1)+'\n')
        prev=OLD/'records'/f'{aid}.json'
        if r['origin_audit_id'] and prev.exists():
            b=prev.read_bytes();(OUT/'reused_originals'/f'{aid}.json').write_bytes(b)
            q=json.loads(b);q.update(review_origin='reused_verified_previous_visual_review',source_sha256=r['result_sha256'],
                reused_review_sha256=hashlib.sha256(b).hexdigest())
            (OUT/'records'/f'{aid}.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n');reuse+=1
        else:pending.append(r)
    assert len(rows)==5378 and len({r['canonical_sample_key'] for r in rows})==len(rows)
    # Balance by actual image workload; independent owned queues.
    pending.sort(key=lambda r:(r['source_version']!='V2_increment',-r['crops'],r['audit_id']))
    queues=[[] for _ in range(6)];loads=[0]*6
    for r in pending:
        w=min(range(6),key=lambda i:loads[i]);queues[w].append(r['audit_id']);loads[w]+=r['crops']+1
    for w,ids in enumerate(queues,1):(OUT/'queues'/f'Q{w}.json').write_text(json.dumps({'reviewer':f'v2_worker_{w}','audit_ids':ids},indent=2)+'\n')
    savejl(OUT/'MANIFEST.jsonl',rows);savejl(OUT/'V2_ALTERNATIVES.jsonl',alternatives)
    savejl(OUT/'V2_NOT_INCLUDED_SOME_USELESS.jsonl',[r for r in candidates if r.get('crops_informative')!='all_useful'])
    st={'V1_paths':len(baseline),'CC_KEEP_paths':len(candidates),'CC_all_useful_paths':len(priority),
        'CC_all_useful_tasks':len(bykey),'new_tasks_added':len(new),'same_task_paths_held':len(alternatives),
        'merged_candidate_T':len(rows),'merged_candidate_Q':len(rows),'merged_candidate_G':len({g for r in rows for g in r['canonical_groups']}),
        'reused_visual_reviews':reuse,'pending_visual_reviews':len(pending),'queue_sizes':[len(q) for q in queues],
        'queue_view_counts':loads,'source_manifest_sha256':sha(OUT/'MANIFEST.jsonl'),
        'all_useful_interpretation':'new detail, binding, comparison, informative negative or justified miss/switch; not crop count or every crop hitting target'}
    (OUT/'PREFLIGHT.json').write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n');print(json.dumps(st,ensure_ascii=False))

if __name__=='__main__':main()
