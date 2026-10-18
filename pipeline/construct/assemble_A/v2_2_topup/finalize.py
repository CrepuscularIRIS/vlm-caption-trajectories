"""Release a separate additive V2.2 after visual review; never mutate frozen V2 or V2.1."""
import collections
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path('${ICML_DIR}'); OUT=Path(__file__).resolve().parent
BASE=ROOT/'obs/benchmarks/sft_v2_1_keep_20261008'
RUN=ROOT/'obs/runs/sft_v2_topup_k3_20261008_r2'
DEST=ROOT/'obs/benchmarks/sft_v2_2_keep_20261009'
GOOD_GAINS={'new_detail','binding_context','comparison_evidence','informative_negative','honest_miss'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def savejl(p,x):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))

def review_errors(ix,r,packet):
    errors=[]; aid=ix['audit_id']; views=[v['view_id'] for v in packet['views']]
    if r.get('audit_id')!=aid:errors.append('id')
    if r.get('verdict') not in {'KEEP','WEAK','BAD','UNRESOLVED'}:errors.append('verdict')
    if sorted(r.get('viewed_ids',[]))!=sorted(views):errors.append('view_coverage')
    ca=r.get('crop_assessment',[])
    if len(ca)!=ix['crops'] or {c['returned_view'] for c in ca}!=set(views[1:]):errors.append('crop_coverage')
    if {c['after_step'] for c in ca}!=set(range(1,ix['crops']+1)):errors.append('crop_steps')
    for issue in r.get('issues',[]):
        quote=issue.get('quote','')
        if not quote:continue
        text=packet['question'] if issue.get('quote_source')=='question' else next((s['raw_text'] for s in packet['steps'] if s['step']==issue.get('step')),'')
        if quote not in text:errors.append('nonverbatim_quote:'+quote)
    if not (OUT/'blind_notes'/f'{aid}.json').exists():errors.append('missing_blind_note')
    if r.get('verdict')=='KEEP':
        if r.get('answer_assessment')!='supported' or r.get('binding_assessment')!='supported':errors.append('KEEP_semantics')
        if r.get('recommendation')!='retain' or any(c['gain'] not in GOOD_GAINS for c in ca):errors.append('KEEP_action_value')
        if any(i.get('severity')=='major' for i in r.get('issues',[])):errors.append('KEEP_major_issue')
    return errors

def main():
    ix=json.loads((OUT/'PRIVATE_INDEX.json').read_text())
    assert len(ix)==88 and len({x['audit_id'] for x in ix})==88
    records={p.stem:json.loads(p.read_text()) for p in (OUT/'records').glob('*.json')}
    parent_by_item={x['item_id']:x for x in json.loads((RUN/'SELECTION_PRIVATE.json').read_text())}
    assert set(records)=={x['audit_id'] for x in ix}, 'missing or extra reviews'
    errors={}
    for row in ix:
        aid=row['audit_id'];packet=json.loads((OUT/'packets'/f'{aid}.json').read_text())
        if aid not in records:errors[aid]=['missing_record'];continue
        e=review_errors(row,records[aid],packet)
        if e:errors[aid]=e
    save(OUT/'REVIEW_VALIDATION.json',dict(reviewed=len(records),expected=88,errors=errors))
    if errors:raise SystemExit(json.dumps(errors,ensure_ascii=False))
    assert all((OUT/'queues'/f'Q{i}_DONE.json').exists() for i in range(1,6))
    assert (OUT/'queues/Q6_ROOT_DONE.json').exists() and (OUT/'queues/Q6_EXTRA_DONE.json').exists()
    q6=json.loads((OUT/'queues/Q6.json').read_text())['audit_ids']
    assert all((OUT/'records'/f'{aid}.json').exists() for aid in q6)
    checks={x['audit_id']:x for x in json.loads((OUT/'EXPORT_CHECKS.json').read_text())}
    lf=json.loads((OUT/'cpu_candidate_check/LF_CHECK.json').read_text())
    mask=json.loads((OUT/'CPU_MASK_CHECK.json').read_text())
    source_data=OUT/'cpu_candidate_check/caption_final_candidates.json'
    assert lf['samples']==lf['passed']==88 and mask['status']=='PASS'
    assert sha(source_data)==mask['raw_dataset_sha256']
    newdata=json.loads(source_data.read_text())
    base_data=json.loads((BASE/'caption_sft_v2_1_keep.json').read_text());base_manifest=jl(BASE/'MANIFEST.jsonl')
    base_lf=json.loads((BASE/'LF_CHECK.json').read_text())
    assert len(base_data)==len(base_manifest)==3979
    assert sha(BASE/'caption_sft_v2_1_keep.json')==base_lf['final_dataset_sha256']
    manifest=copy.deepcopy(base_manifest);data=copy.deepcopy(base_data);joined=[];increment=[]
    keys={m['canonical_sample_key'] for m in manifest};groups={g for m in manifest for g in m['canonical_groups']}
    new_lf=[]
    for row in ix:
        aid=row['audit_id'];rp=Path(row['result']);assert sha(rp)==row['result_sha256']
        raw=json.loads(rp.read_text());ro=raw['rollout'];check=checks[aid];r=records[aid]
        assert not check['errors'] and check['result_sha256']==row['result_sha256']
        ex=newdata[check['idx']]
        assert ex['conversations']==ro['export']['conversations']
        assert ex['images']==[str((ROOT/p).resolve()) for p in ro['export']['images']]
        assert [sha(p) for p in ex['images']]==[v['sha256'] for v in ro['views']]
        m=dict(audit_id=aid,item_id=row['item_id'],canonical_sample_key=row['item']['actual_sample_key'],
            canonical_groups=[row['item']['canonical_group_id']],model='K3',bench=row['item']['bench'],material=row['item']['material'],
            source_version='V2.2_topup',result_path=str(rp),result_sha256=row['result_sha256'],crops=row['crops'],steps=row['steps'],
            review_record_sha256=sha(OUT/'records'/f'{aid}.json'))
        full=dict(m,verdict=r['verdict'],review=r,lf_check=lf['rows'][check['idx']],
                  disposition='core_ready' if r['verdict']=='KEEP' else 'not_selected',human_acceptance=False,
                  previous_audit_id=parent_by_item[row['item_id']]['audit_id'],previous_verdict=parent_by_item[row['item_id']]['prior_verdict'])
        joined.append(full)
        if r['verdict']!='KEEP':continue
        assert m['canonical_sample_key'] not in keys and not (set(m['canonical_groups']) & groups)
        assert raw['score']['score']==1 and raw['cpu']['passed'] and raw['item']['split']=='train'
        keys.add(m['canonical_sample_key']);groups.update(m['canonical_groups'])
        increment.append(full);manifest.append(m);data.append(ex)
        new_lf.append(dict(lf['rows'][check['idx']],audit_id=aid))
    DEST.mkdir(exist_ok=True)
    if (DEST/'caption_sft_v2_2_keep.json').exists():
        assert json.loads((DEST/'caption_sft_v2_2_keep.json').read_text())==data,'Refuse changed repeat export'
    save(DEST/'caption_sft_v2_2_keep.json',data);savejl(DEST/'MANIFEST.jsonl',manifest)
    info=json.loads((BASE/'dataset_info.json').read_text())['caption_sft_v2_1_keep']
    info=copy.deepcopy(info);info['file_name']='caption_sft_v2_2_keep.json';save(DEST/'dataset_info.json',{'caption_sft_v2_2_keep':info})
    savejl(OUT/'RESULTS.jsonl',joined);savejl(OUT/'INCREMENT_KEEP.jsonl',increment)
    model=collections.defaultdict(lambda:collections.Counter())
    for m in manifest:
        model[m['model']]['total']+=1
        model[m['model']]['direct' if m['crops']==0 else 'one_crop' if m['crops']==1 else 'two_plus_crops']+=1
    flf=copy.deepcopy(base_lf['rows'])+new_lf
    length={'max_tokens':max(x['tokens'] for x in flf),'mean_tokens':round(sum(x['tokens'] for x in flf)/len(flf),2),
            'max_steps':max(x['steps'] for x in manifest),'mean_steps':round(sum(x['steps'] for x in manifest)/len(manifest),3)}
    stats=dict(version='V2.2',status='complete_model_reviewed_not_human_gold',baseline=dict(T=3979,Q=3979,G=3629),
        generation=dict(K3_complete=111,K3_correct=90,K3_CPU_correct=88,K3_HTTP_requests=292,technical_errors=0,all_attempts_settled=True),
        process_reviewed=88,actual_views=200,quality=dict(collections.Counter(x['verdict'] for x in joined)),
        gate_excluded_not_process_reviewed=23,increment_KEEP=len(increment),
        increment_behavior=dict(collections.Counter('direct' if x['crops']==0 else 'one_crop' if x['crops']==1 else 'two_plus_crops' for x in increment)),
        final_T_Q_G=dict(T=len(data),Q=len(keys),G=len(groups)),needed_for_over_4000=max(0,4001-len(data)),
        by_model=dict(model),by_benchmark=dict(collections.Counter(m['bench'] for m in manifest)),
        final_behavior=dict(collections.Counter('direct' if x['crops']==0 else 'one_crop' if x['crops']==1 else 'two_plus_crops' for x in manifest)),
        final_lengths=length,new_CPU=dict(samples=88,passed=88,max_tokens=lf['max_tokens'],exact_mask_passed=88),
        final_dataset_sha256=sha(DEST/'caption_sft_v2_2_keep.json'),baseline_dataset_sha256=base_lf['final_dataset_sha256'],
        human_acceptance=False,training=False,GPU=False,additional_model_calls_for_review=0,
        limitations=['model review, not human gold','full-path review is not temporally isolated prefix verification',
                     'different tasks, not a model ranking','student-resolution evidence legibility and training utility untested'])
    stats['review_by_prior_verdict']={v:dict(collections.Counter(x['verdict'] for x in joined if x['previous_verdict']==v)) for v in ('KEEP','WEAK','BAD','UNRESOLVED')}
    stats['review_by_crop_bin']={v:dict(collections.Counter(x['verdict'] for x in joined if ('direct' if x['crops']==0 else 'one_crop' if x['crops']==1 else 'two_plus_crops')==v)) for v in ('direct','one_crop','two_plus_crops')}
    stats['review_by_benchmark']={v:dict(collections.Counter(x['verdict'] for x in joined if x['bench']==v)) for v in sorted({x['bench'] for x in joined})}
    stats['increment_mentor_behaviors']=dict(collections.Counter(b for x in increment for b in x['review']['mentor_behaviors']))
    stats['increment_crop_gains']=dict(collections.Counter(c['gain'] for x in increment for c in x['review']['crop_assessment']))
    stats['exactly_one_path_per_task']=len(data)==len(keys)
    stats['reviewers']='five native Codex workers plus root; Q4 also handled nine Q6 tasks; one model-system review per task, no external judge'
    save(OUT/'STATS.json',stats);save(DEST/'STATS.json',stats)
    save(DEST/'LF_CHECK.json',dict(status='PASS',samples=len(data),passed=len(data),basis='unchanged verified V2.1 plus unchanged verified K3 r2 subset',
        baseline_dataset_sha256=base_lf['final_dataset_sha256'],baseline_LF_CHECK_sha256=sha(BASE/'LF_CHECK.json'),
        topup_dataset_sha256=sha(source_data),topup_LF_CHECK_sha256=sha(OUT/'cpu_candidate_check/LF_CHECK.json'),
        topup_mask_CHECK_sha256=sha(OUT/'CPU_MASK_CHECK.json'),final_dataset_sha256=stats['final_dataset_sha256'],rows=flf,no_GPU=True,no_training=True))
    # Public output is staged locally first: no raw questions, pixels or API receipts.
    spec=importlib.util.spec_from_file_location('v2_public',ROOT/'reports/sft_v2_merged_20261008/export_release.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    preview=OUT/'public_preview';preview.mkdir(exist_ok=True);visible=[];table=[]
    for row in joined:
        rp=Path(row['result_path']);ro=json.loads(rp.read_text())['rollout']
        public={k:row[k] for k in ('audit_id','item_id','canonical_sample_key','canonical_groups','model','bench','material','source_version','crops','steps','verdict','disposition','result_sha256','review_record_sha256')}
        public.update(review=mod.public_review(row['review']),
            lf_check=row['lf_check'],acceptance='model_reviewed_not_human_gold',previous_audit_id=row['previous_audit_id'],previous_verdict=row['previous_verdict'])
        table.append(public)
        if row['verdict']!='KEEP':continue
        visible.append(dict(public,views=[dict(view_id=v['view_id'],parent=v['parent'],sha256=v['sha256'],shown_px=v['shown_px'],
            fov_original=v['fov_original'],asset_included=False) for v in ro['views']],
            assistant_turns=[s['raw_text'] for s in ro['steps']],text_format='visible_caption_raw',
            step_context=[dict(step=s['step'],available_before=s['available_before']) for s in ro['steps']],terminal_answer=ro['answer'],protocol='caption_v5.1'))
    savejl(preview/'v2_2_increment_keep_20261009.jsonl',visible)
    savejl(preview/'v2_2_topup_review_table_20261009.jsonl',table)
    save(preview/'v2_2_summary_20261009.json',stats)
    with (preview/'v2_2_topup_review_table_20261009.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['audit_id','item_id','model','bench','crops','steps','verdict','advice']);w.writeheader()
        for x in table:w.writerow({**{k:x[k] for k in ['audit_id','item_id','model','bench','crops','steps','verdict']},'advice':x['review']['summary_zh']})
    print(json.dumps(stats,ensure_ascii=False))

if __name__=='__main__':main()
