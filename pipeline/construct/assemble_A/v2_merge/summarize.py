"""Join independent reviews and history without changing the frozen selection.

Missing rows remain NOT_REVIEWED. This script performs no visual grading.
"""
import collections
import csv
import datetime
import hashlib
import json
import re
from pathlib import Path

OUT=Path(__file__).resolve().parent
GRADES=('KEEP','WEAK','BAD','UNRESOLVED','NOT_REVIEWED')
MODELS=('Luna','Sol 5.6','GLM','Grok','Astra','Gemini','K3')
BEHAVIORS=('locate_then_detail','cross_region_compare_or_bind','honest_miss_then_switch','direct_when_sufficient')

def jl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def savejl(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
def norm(s):return re.sub(r'\s+',' ',str(s)).strip()
def t_q_g(rows):return {'T':len(rows),'Q':len({r['canonical_sample_key'] for r in rows}),
    'G':len({g for r in rows for g in r['canonical_groups']})}

def validate(r,p):
    errors=[]
    required={'audit_id','reviewer','verdict','viewed_ids','answer_assessment','binding_assessment',
              'mentor_behaviors','behavior_evidence','crop_assessment','issues','useful_multistep','recommendation','summary_zh'}
    if required-set(r):errors.append('missing_fields:'+','.join(sorted(required-set(r))))
    views={v['view_id'] for v in p['views']}
    if r.get('audit_id')!=p['audit_id']:errors.append('audit_id_mismatch')
    if r.get('verdict') not in GRADES[:4]:errors.append('invalid_verdict')
    seen=r.get('viewed_ids',[])
    if set(seen)!=views or len(seen)!=len(views):errors.append('view_coverage_mismatch')
    crops=r.get('crop_assessment',[])
    if {v.get('returned_view') for v in crops} != views-{'original_image'} or len(crops)!=len(views)-1:
        errors.append('crop_coverage_mismatch')
    for c in crops:
        if c.get('gain') not in {'new_detail','binding_context','comparison_evidence','informative_negative','honest_miss','no_new_evidence','uncertain'}:
            errors.append('invalid_crop_gain')
        n=c.get('after_step')
        step=next((s for s in p['steps'] if s['step']==n),None)
        if step is None or step['parsed'].get('kind')!='grounding' or step.get('observation')!=c.get('returned_view'):
            errors.append('crop_step_mismatch')
    for i in r.get('issues',[]):
        if i.get('severity') not in {'minor','major','uncertain'}:errors.append('invalid_issue_severity')
        step=next((s for s in p['steps'] if s['step']==i.get('step')),None)
        if step is None:errors.append('issue_step_not_found');continue
        quote=norm(i.get('quote',''))
        anchor=p['question'] if i.get('quote_source')=='question' else step['parsed']['text']
        if quote and quote not in norm(anchor):errors.append('issue_quote_not_exact')
        if i.get('view') not in views:errors.append('issue_view_not_found')
    if r.get('answer_assessment') not in {'supported','contradicted','uncertain'}:errors.append('invalid_answer_assessment')
    if r.get('binding_assessment') not in {'supported','contradicted','uncertain'}:errors.append('invalid_binding_assessment')
    if r.get('recommendation') not in {'retain','prefer_existing_alternative','hold_for_visual_adjudication','recovery_only'}:errors.append('invalid_recommendation')
    if r.get('useful_multistep') not in {'yes','no','uncertain','not_applicable'}:errors.append('invalid_useful_multistep')
    if any(b not in BEHAVIORS for b in r.get('mentor_behaviors',[])):errors.append('invalid_mentor_behavior')
    step_ids={s['step'] for s in p['steps']}
    for e in r.get('behavior_evidence',[]):
        if e.get('behavior') not in BEHAVIORS or not set(e.get('steps',[]))<=step_ids or not set(e.get('views',[]))<=views:
            errors.append('invalid_behavior_evidence_anchor')
    return sorted(set(errors))

def aggregate(rows):
    out={'paths':len(rows),'quality':{g:sum(r['verdict']==g for r in rows) for g in GRADES},
         'direct':sum(r['crops']==0 for r in rows),'one_crop':sum(r['crops']==1 for r in rows),
         'two_plus_crops':sum(r['crops']>=2 for r in rows)}
    for pool in ('core_ready','weak_reserve','review_hold','excluded','pending'):
        out[pool]=sum(r['disposition']==pool for r in rows)
    core=[r for r in rows if r['disposition']=='core_ready']
    out['core_direct']=sum(r['crops']==0 for r in core)
    out['core_one_crop']=sum(r['crops']==1 for r in core)
    out['core_two_plus_crops']=sum(r['crops']>=2 for r in core)
    return out

def main():
    manifest=jl(OUT/'MANIFEST.jsonl'); history={r['audit_id']:r for r in jl(OUT/'HISTORY_ENRICHED.jsonl')}
    checks={r['audit_id']:r for r in jl(OUT/'EXPORT_CHECKS.jsonl')}
    lfpath=OUT/'cpu_candidate_check/LF_CHECK.json'
    lf=json.loads(lfpath.read_text()) if lfpath.exists() else None
    lfrows={r['idx']:r for r in lf['rows']} if lf else {}
    maskpath=OUT/'CPU_MASK_CHECK.json'
    maskcheck=json.loads(maskpath.read_text()) if maskpath.exists() else None
    mask_ok=bool(maskcheck and maskcheck['status']=='PASS' and maskcheck['exact_mask_and_target_passed']==len(manifest)
        and maskcheck['raw_dataset_sha256']==sha(OUT/'cpu_candidate_check/caption_final_candidates.json'))
    rows=[]; format_errors=[]; blockers=[]
    for m in manifest:
        aid=m['audit_id']; f=OUT/'records'/f'{aid}.json'; r=None; errs=[]
        if f.exists():
            try:r=json.loads(f.read_text())
            except json.JSONDecodeError:errs=['invalid_record_json']
        p=json.loads((OUT/'packets'/f'{aid}.json').read_text())
        if r:errs+=validate(r,p)
        grade=r.get('verdict','NOT_REVIEWED') if r else 'NOT_REVIEWED'
        if errs:format_errors.append({'audit_id':aid,'errors':errs,'reviewer':r.get('reviewer') if r else None})
        quality_flags=[]
        if r:
            if grade=='KEEP' and (r.get('answer_assessment')!='supported' or r.get('binding_assessment')!='supported'):
                quality_flags.append('KEEP_with_uncertain_or_contradicted_core')
            if grade=='KEEP' and any(i.get('severity') in ('major','uncertain') for i in r.get('issues',[])):
                quality_flags.append('KEEP_with_major_or_uncertain_issue')
            if grade=='KEEP' and m['crops']>=2 and r.get('useful_multistep')!='yes':
                quality_flags.append('KEEP_without_useful_multistep')
            if grade=='KEEP' and any(c.get('gain') in ('no_new_evidence','uncertain') for c in r.get('crop_assessment',[])):
                quality_flags.append('KEEP_with_crop_not_confirmed_useful')
        tech=checks[aid]; lfr=lfrows.get(tech['idx'])
        lfok=bool(mask_ok and lfr and not lfr['truncated'] and lfr['think_in_target']==lfr['gpt_turns']
             and lfr['actions_in_target']==lfr['gpt_turns'] and not lfr['receipt_leak']
             and lfr['images']==lfr['vision_blocks'] and lf['samples']==len(manifest))
        disposition='pending' if not r else 'review_hold'
        recommendation=r.get('recommendation') if r else None
        if r and not errs and not tech['errors'] and not quality_flags:
            if grade=='BAD':disposition='excluded'
            elif grade=='KEEP' and recommendation=='retain' and lfok:disposition='core_ready'
            elif grade=='WEAK':disposition='weak_reserve'
        if quality_flags:blockers.append({'audit_id':aid,'flags':quality_flags})
        h=history[aid]
        row={**m,'verdict':grade,'review':r,'review_format_errors':errs,'quality_flags':quality_flags,
             'technical_errors':tech['errors'],'lf_check':lfr,'lf_passed':lfok,'disposition':disposition,
             'review_record_sha256':sha(f) if f.exists() else None,'historical_reviews':h['reviews']}
        rows.append(row)
    savejl(OUT/'RESULTS.jsonl',rows)
    savejl(OUT/'CORE_READY.jsonl',[r for r in rows if r['disposition']=='core_ready'])
    savejl(OUT/'WEAK_RESERVE.jsonl',[r for r in rows if r['disposition']=='weak_reserve'])
    savejl(OUT/'FOLLOWUP.jsonl',[r for r in rows if r['disposition'] in ('review_hold','excluded')])
    (OUT/'VALIDATION_ISSUES.json').write_text(json.dumps({'format':format_errors,'semantic_field_flags':blockers},ensure_ascii=False,indent=2)+'\n')
    reviewed=[r for r in rows if r['review']];core=[r for r in rows if r['disposition']=='core_ready']
    status='complete' if len(reviewed)==len(rows) and not format_errors else 'in_progress'
    stats={'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,
       'candidate_T_Q_G':t_q_g(rows),'reviewed_paths':len(reviewed),'pending_paths':len(rows)-len(reviewed),
       'reviewed_total_views':sum(len(r['review']['viewed_ids']) for r in reviewed),
       'reused_reviews':sum(r['review'].get('review_origin','').startswith('reused_') for r in reviewed),
       'all':aggregate(rows),'by_model':{m:aggregate([r for r in rows if r['model']==m]) for m in MODELS},
       'by_benchmark':{m:aggregate([r for r in rows if r['bench']==m]) for m in sorted({r['bench'] for r in rows})},
       'by_crop_bin':{b:aggregate([r for r in rows if ('0' if r['crops']==0 else '1' if r['crops']==1 else '2+')==b]) for b in ('0','1','2+')},
       'core_ready_T_Q_G':t_q_g(core),'core_mentor_behaviors':dict(collections.Counter(b for r in core for b in r['review']['mentor_behaviors'])),
       'reviewer_recommendations':dict(collections.Counter(r['review']['recommendation'] for r in reviewed)),
       'record_format_errors':len(format_errors),'semantic_field_flags':len(blockers),
       'historical_advice_available':sum(bool(r['historical_reviews']) for r in rows),
       'human_acceptance':False,'training_run':False,'external_model_calls':0,
       'source_manifest_sha256':json.loads((OUT/'PREFLIGHT.json').read_text())['source_manifest_sha256'],
       'cpu_mask_check':maskcheck,
       'lf_summary':{k:v for k,v in lf.items() if k!='rows'} if lf else None}
    (OUT/'STATS.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    fields=['audit_id','item_id','model','bench','material','crops','steps','verdict','disposition',
            'new_advice','mentor_behaviors','historical_review_source','historical_reviewer','historical_advice','result_sha256']
    with (OUT/'REVIEW_TABLE.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            d={k:r.get(k) for k in fields}; d.update(new_advice=(r['review'] or {}).get('summary_zh','未核'),
              mentor_behaviors=';'.join((r['review'] or {}).get('mentor_behaviors',[])),historical_review_source=r['review_source'],
              historical_reviewer=';'.join(sorted({h.get('reviewer_attribution','unknown') for h in r['historical_reviews']})),
              historical_advice='\n'.join(str(h.get('issue') or h.get('why') or '') for h in r['historical_reviews']))
            w.writerow(d)
    print(json.dumps({k:stats[k] for k in ('status','reviewed_paths','pending_paths','record_format_errors','semantic_field_flags','all')},ensure_ascii=False))

if __name__=='__main__':main()
