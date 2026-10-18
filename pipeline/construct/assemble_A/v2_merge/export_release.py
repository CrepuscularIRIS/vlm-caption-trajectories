"""Export only strict KEEP paths after all frozen merged candidates are reviewed."""
import argparse
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path('${ICML_DIR}');OUT=Path(__file__).resolve().parent
PUB=ROOT/'publications/vlm-caption-trajectories'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def savejl(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
def clean(x):
    if isinstance(x,str):return x.replace(str(ROOT)+'/','').replace('${HOME}/','[private-home]/')
    if isinstance(x,list):return [clean(y) for y in x]
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    return x
def public_review(r):
    if not r:return None
    d={k:r.get(k) for k in ('reviewer','verdict','viewed_ids','answer_assessment','binding_assessment',
       'mentor_behaviors','behavior_evidence','crop_assessment','issues','useful_multistep','recommendation','summary_zh','review_origin','external_verification','review_limitations')}
    d=copy.deepcopy(d)
    for i in d.get('issues') or []:
        if i.get('quote_source')=='question':
            i['quote_sha256']=hashlib.sha256(i.get('quote','').encode()).hexdigest()
            i['quote']='[Source question retained in licensed local data.]'
    return clean(d)
def public_history(rs):
    return [clean({k:r[k] for k in ('verdict','final','issue','why','why_observations_needed',
        'binding','evidence','action_fit','redundancy','imitate','crops_informative','observed_behavior',
        'answerability','recovery_candidate','reviewer_attribution','attribution_evidence') if k in r}) for r in rs]

def public_stats(stats):
    allowed='updated_at status candidate_T_Q_G reviewed_paths pending_paths reviewed_total_views reused_reviews all by_model by_benchmark by_crop_bin core_ready_T_Q_G core_mentor_behaviors reviewer_recommendations record_format_errors semantic_field_flags historical_advice_available human_acceptance training_run external_model_calls source_manifest_sha256 core_length core_crop_histogram core_step_histogram core_by_origin core_by_material core_by_benchmark by_origin_disposition core_groups_with_multiple_questions core_crop_gains core_useful_crop_fraction reviewed_recovery_only'.split()
    d={k:stats[k] for k in allowed if k in stats}
    mask_allowed='status samples exact_mask_and_target_passed cache_sha256 raw_dataset_sha256 check_script_sha256 runtime_versions processor_max_pixels processor_min_pixels train_on_prompt mask_history saved_views saved_views_above_pixel_budget pixel_budget_note no_GPU no_training no_image_reprocessing'.split()
    d['cpu_mask_check']={k:stats['cpu_mask_check'][k] for k in mask_allowed if k in stats['cpu_mask_check']}
    d['lf_summary']={k:stats['lf_summary'][k] for k in ('samples','passed','max_tokens','mean_tokens','cutoff')}
    return clean(d)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--publish-files',action='store_true');a=ap.parse_args()
    stats=json.loads((OUT/'STATS.json').read_text());rows=jl(OUT/'RESULTS.jsonl')
    if stats['status']!='complete' or stats['pending_paths'] or stats['record_format_errors']:
        raise SystemExit('Refuse final release while visual reviews are incomplete or malformed.')
    core=[r for r in rows if r['disposition']=='core_ready']
    assert len(core)==len({r['canonical_sample_key'] for r in core})==stats['core_ready_T_Q_G']['T']
    assert all(r['verdict']=='KEEP' and r['lf_passed'] and not r['technical_errors'] and not r['quality_flags'] for r in core)
    assert stats['cpu_mask_check']['status']=='PASS'
    candidate_path=OUT/'cpu_candidate_check/caption_final_candidates.json'
    assert sha(candidate_path)==stats['cpu_mask_check']['raw_dataset_sha256'],'CPU-checked dataset changed'
    assert json.loads((OUT/'CPU_MASK_CHECK.json').read_text())==stats['cpu_mask_check'],'CPU mask receipt changed'
    dataset=json.loads(candidate_path.read_text())
    current_lf=json.loads((OUT/'cpu_candidate_check/LF_CHECK.json').read_text())
    assert {k:v for k,v in current_lf.items() if k!='rows'}==stats['lf_summary'],'LF receipt changed'
    lf_rows={v['idx']:v for v in current_lf['rows']}
    checks={r['audit_id']:r for r in jl(OUT/'EXPORT_CHECKS.jsonl')}
    dest=ROOT/'obs/benchmarks/sft_v2_merged_keep_20261008';dest.mkdir(exist_ok=True)
    local=[];manifest=[];visible=[];table=[]
    for r in rows:
        rp=Path(r['result_path']);assert sha(rp)==r['result_sha256']
        assert checks[r['audit_id']]['source_sha256']==r['result_sha256']
        assert checks[r['audit_id']]['errors']==r['technical_errors'],'Protocol check changed'
        review_path=OUT/'records'/f"{r['audit_id']}.json"
        assert sha(review_path)==r['review_record_sha256'],'Visual review changed after summary'
        assert json.loads(review_path.read_text())==r['review'],'Visual review content mismatch'
        assert lf_rows[checks[r['audit_id']]['idx']]==r['lf_check'],'LF row mismatch'
        packet=json.loads((OUT/'packets'/f"{r['audit_id']}.json").read_text())
        public={k:r[k] for k in ('audit_id','item_id','canonical_sample_key','canonical_groups','model','bench','material',
           'source_version','crops','steps','verdict','disposition','result_sha256','review_source','review_record_sha256')}
        public.update(source_result=str(rp.relative_to(ROOT)),question_sha256=hashlib.sha256(packet['question'].encode()).hexdigest(),
            review=public_review(r['review']),historical_reviews=public_history(r['historical_reviews']),
            technical_errors=r['technical_errors'],review_format_errors=r['review_format_errors'],quality_flags=r['quality_flags'],
            lf_passed=r['lf_passed'],lf_check=r['lf_check'],acceptance='model_reviewed_not_human_gold')
        table.append(public)
        if r['disposition']!='core_ready':continue
        rec=dataset[checks[r['audit_id']]['idx']]
        source=json.loads(rp.read_text())['rollout']['export']
        assert rec['conversations']==source['conversations']
        assert [str((ROOT/p).resolve()) for p in source['images']]==rec['images'],'Source image order mismatch'
        views=[]
        for v,path in zip(packet['views'],rec['images']):
            vp=Path(path);assert sha(vp)==v['sha256']
            views.append({**{k:v.get(k) for k in ('view_id','parent','sha256','shown_px','fov_original')},
                'source_path':str(vp.relative_to(ROOT)),'asset_included':False})
        assistants=[c['value'] for c in rec['conversations'] if c['from']=='gpt']
        assert len(assistants)==r['steps'] and len(views)==r['crops']+1
        visible.append({**public,'views':views,'assistant_turns':assistants,
            'step_context':[{'step':s['step'],'available_before':s['available_before']} for s in packet['steps']],
            'terminal_answer':packet['answer'],'protocol':'caption_v5.1',
            'assets_note':'Generated visible captions/actions only; source question and images require lawful local access.'})
        local.append(rec);manifest.append({k:r[k] for k in ('audit_id','item_id','canonical_sample_key','canonical_groups','model','bench',
            'material','source_version','result_path','result_sha256','crops','steps','review_record_sha256')})
    name='caption_sft_v2_keep'
    (dest/f'{name}.json').write_text(json.dumps(local,ensure_ascii=False))
    info=json.loads((OUT/'cpu_candidate_check/dataset_info.json').read_text())['caption_final_candidates']
    info['file_name']=f'{name}.json';(dest/'dataset_info.json').write_text(json.dumps({name:info},indent=2)+'\n')
    savejl(dest/'MANIFEST.jsonl',manifest)
    check={'status':'PASS','samples':len(local),'passed':len(local),'basis':'unchanged exact subset of full candidate CPU preprocessing',
      'full_candidate_samples':stats['lf_summary']['samples'],'full_candidate_passed':stats['lf_summary']['passed'],
      'candidate_dataset_sha256':sha(OUT/'cpu_candidate_check/caption_final_candidates.json'),
      'full_LF_CHECK_sha256':sha(OUT/'cpu_candidate_check/LF_CHECK.json'),
      'exact_mask_check_sha256':sha(OUT/'CPU_MASK_CHECK.json'),
      'final_dataset_sha256':sha(dest/f'{name}.json'),'rows':[dict(r['lf_check'],audit_id=r['audit_id']) for r in core],
      'no_training':True,'no_GPU':True,'human_acceptance':False}
    (dest/'LF_CHECK.json').write_text(json.dumps(check,ensure_ascii=False,indent=1)+'\n')
    (dest/'STATS.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    # Public text artifacts never include original question text, source images,
    # hidden reasoning, credentials, raw API receipts, or private conversations.
    target=PUB/'data' if a.publish_files else OUT/'public_preview';target.mkdir(exist_ok=True)
    savejl(target/'v2_merged_review_table_20261008.jsonl',table)
    savejl(target/'v2_merged_keep_trajectories_20261008.jsonl',visible)
    (target/'v2_merged_summary_20261008.json').write_text(json.dumps(public_stats(stats),ensure_ascii=False,indent=2)+'\n')
    header=['audit_id','item_id','model','bench','source_version','crops','steps','verdict','disposition','current_advice','historical_reviewers','historical_advice']
    with (target/'v2_merged_review_table_20261008.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=header);w.writeheader()
        for r in table:
            d={k:r.get(k) for k in header};d.update(current_advice=r['review']['summary_zh'],
              historical_reviewers=';'.join(sorted({str(h.get('reviewer_attribution','unknown')) for h in r['historical_reviews']})),
              historical_advice='\n'.join(str(h.get('issue') or h.get('why') or '') for h in r['historical_reviews']));w.writerow(d)
    if a.publish_files:(PUB/'docs/16_v2_merged_trajectories_20261008.md').write_text((OUT/'REPORT.zh.md').read_text())
    print(json.dumps({'export_dir':str(dest),'core_T_Q_G':stats['core_ready_T_Q_G'],'public_files_written':a.publish_files},ensure_ascii=False))

if __name__=='__main__':main()
