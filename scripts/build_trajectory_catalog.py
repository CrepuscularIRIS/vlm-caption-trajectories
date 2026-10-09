#!/usr/bin/env python3
"""Build a path-free public catalog from frozen, previously published records.
No model calls, label edits, pixel copies, private source files or training.
"""
import argparse,collections,csv,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODELS=['Luna','Sol 5.6','GLM','Grok','Astra','Gemini','K3']
QUALITY=['KEEP','WEAK','BAD','UNRESOLVED']
SOURCES=[('V2','v2_merged_keep_trajectories_20261008.jsonl','v2_merged_review_table_20261008.jsonl'),('V2.1','v2_1_increment_keep_20261008.jsonl','v2_1_topup_review_table_20261008.jsonl'),('V2.2','v2_2_increment_keep_20261009.jsonl','v2_2_topup_review_table_20261009.jsonl')]
def read(name):return [json.loads(x) for x in (ROOT/'data'/name).read_text().splitlines() if x.strip()]
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,separators=(',',':'))+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(xs):
 a=sorted(xs);return dict(n=len(a),total=sum(a),mean=round(statistics.mean(a),3),median=statistics.median(a),p90=a[math.ceil(.9*len(a))-1],max=max(a))
def aggregate(rows):
 return dict(paths=len(rows),direct=sum(r['crops']==0 for r in rows),one_crop=sum(r['crops']==1 for r in rows),two_plus_crops=sum(r['crops']>=2 for r in rows),steps=metric([r['steps'] for r in rows]),processed_tokens=metric([r['lf_check']['tokens'] for r in rows]),target_tokens=metric([r['lf_check']['target_tokens'] for r in rows]))
def clean_review(r):
 # A strict whitelist excludes local paths, runtime, raw source questions and API receipts.
 keys=['verdict','viewed_ids','answer_assessment','binding_assessment','mentor_behaviors','behavior_evidence','crop_assessment','issues','useful_multistep','recommendation','summary_zh','review_origin','review_limitations','external_verification']
 def clean(x):
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('source_path','source_result','result_path','image_path','question','reasoning_content')}
  if isinstance(x,list):return [clean(v) for v in x]
  if isinstance(x,str):
   import re
   return re.sub(r'/(?:home|data|tmp)/[^\s\"<>]+','[local reference omitted]',x).replace('<think>','<caption>').replace('</think>','</caption>')
  return x
 return clean({k:r.get(k) for k in keys if r.get(k) is not None})
def build():
 final=[];audits=[];phases=[];inputs={}
 for phase,traces,reviews in SOURCES:
  fs=read(traces);rs=read(reviews);final.extend(fs)
  for r in rs:r['_phase']=phase
  audits.extend(rs);phases.append(dict(phase=phase,reviewed=len(rs),quality=dict(collections.Counter(r['verdict'] for r in rs)),exported=len(fs),technical_KEEP_excluded=sum(r['verdict']=='KEEP' for r in rs)-len(fs)))
  for name in [traces,reviews]:inputs[name]=digest(ROOT/'data'/name)
 selected={r['audit_id']:r for r in final};assert len(selected)==4040==len(final)
 assert len({r['audit_id'] for r in audits})==len(audits)==5503
 assert all(r['steps']==r['crops']+1==len(r['assistant_turns'])==len(r['views']) and r['verdict']=='KEEP' for r in final)
 assert all(r['review']['answer_assessment']==r['review']['binding_assessment']=='supported' for r in final)
 assert all(c['gain'] in {'new_detail','binding_context','comparison_evidence','informative_negative','honest_miss'} for r in final for c in r['review']['crop_assessment'])
 summary=dict(version='V2.2',date='2026-10-09',T=len(final),Q=len({r['canonical_sample_key'] for r in final}),G=len({g for r in final for g in r['canonical_groups']}),models=7,reviewed_paths=len(audits),reviewed_unique_tasks=len({r['canonical_sample_key'] for r in audits}),quality=dict(collections.Counter(r['verdict'] for r in audits)),phases=phases,
  lengths=aggregate(final),crop_length=metric([r['crops'] for r in final]),step_histogram=dict(sorted(collections.Counter(r['steps'] for r in final).items())),by_model={m:aggregate([r for r in final if r['model']==m]) for m in MODELS},
  by_source={m:aggregate([r for r in final if r['bench']==m]) for m in sorted({r['bench'] for r in final})},by_material={m:aggregate([r for r in final if r['material']==m]) for m in sorted({r['material'] for r in final})},
  audit_by_model={m:dict(reviewed=sum(r['model']==m for r in audits),quality=dict(collections.Counter(r['verdict'] for r in audits if r['model']==m)),exported=sum(r['model']==m for r in final)) for m in MODELS},
  audit_by_source={m:dict(reviewed=sum(r['bench']==m for r in audits),quality=dict(collections.Counter(r['verdict'] for r in audits if r['bench']==m)),exported=sum(r['bench']==m for r in final)) for m in sorted({r['bench'] for r in final})},
  behaviors=dict(collections.Counter(b for r in final for b in r['review']['mentor_behaviors'])),crop_gains=dict(collections.Counter(c['gain'] for r in final for c in r['review']['crop_assessment'])),
  historical_advice_paths=sum(bool(r.get('historical_reviews')) for r in final),historical_attribution_paths={name:sum(any(h.get('reviewer_attribution')==name for h in r.get('historical_reviews',[])) for r in final) for name in sorted({h.get('reviewer_attribution','unknown') for r in final for h in r.get('historical_reviews',[])})},
  human_gold=False,training_run=False,all_selected_have_process_KEEP=True,unreviewed_selected=0,source_inputs=inputs)
 groups=collections.Counter(g for r in final for g in r['canonical_groups']);summary['groups_with_multiple_tasks']=sum(n>1 for n in groups.values())
 index=[];detail=[]
 for i,r in enumerate(audits):
  rid=r['audit_id'];f=selected.get(rid);rv=clean_review(r['review']);hist=r.get('historical_reviews',[])
  meta={k:r[k] for k in ('audit_id','item_id','model','bench','material','steps','crops','verdict')}
  meta.update(selected=f is not None,phase=r['_phase'],shard=f'data/paths-{i//100:03d}.json',behaviors=rv.get('mentor_behaviors',[]),summary=rv.get('summary_zh',''))
  index.append(meta)
  d=dict(meta,review=rv,historical_reviews=[clean_review({**h,'summary_zh':h.get('issue') or h.get('why'),'review_origin':h.get('reviewer_attribution','historical')}) for h in hist],
         result_sha256=r['result_sha256'],review_record_sha256=r.get('review_record_sha256'),groups=r['canonical_groups'],disposition=r['disposition'],terminal_answer=None,
         public_images=False,lf_check={k:v for k,v in (r.get('lf_check') or {}).items() if k in ['tokens','target_tokens','gpt_turns','images','truncated']})
  if f:
   d.update(assistant_turns=[t.replace('<think>','<caption>').replace('</think>','</caption>') for t in f['assistant_turns']],step_context=f['step_context'],terminal_answer=f['terminal_answer'],
    views=[{k:v[k] for k in ('view_id','parent','sha256','shown_px','fov_original')} for v in f['views']],caption_display='Visible short evidence caption; tag normalized for display, original export unchanged.')
  detail.append(d)
 web=ROOT/'web';save(web/'data/catalog.json',index);save(web/'data/summary.json',summary)
 for i in range(0,len(detail),100):save(web/f'data/paths-{i//100:03d}.json',detail[i:i+100])
 save(ROOT/'data/v2_2_complete_statistics_20261009.json',summary)
 with (ROOT/'data/v2_2_complete_inventory_20261009.csv').open('w',newline='',encoding='utf-8-sig') as fp:
  names=['audit_id','item_id','model','bench','material','steps','crops','verdict','selected','phase'];w=csv.DictWriter(fp,fieldnames=names,lineterminator="\n");w.writeheader();w.writerows({k:r[k] for k in names} for r in index)
 manifest={str(p.relative_to(web)):digest(p) for p in sorted((web/'data').glob('*.json'))};save(web/'catalog_manifest.json',manifest)
 print(json.dumps({'selected':len(final),'reviewed':len(audits),'steps':summary['lengths']['steps'],'shards':math.ceil(len(detail)/100),'bytes':sum(p.stat().st_size for p in (web/'data').glob('*.json'))},ensure_ascii=False))
if __name__=='__main__':build()
