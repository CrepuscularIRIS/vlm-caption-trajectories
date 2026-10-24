import collections,hashlib,json,random,statistics,sys,importlib.util
from pathlib import Path
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--clean',type=Path,required=True);ap.add_argument('--e2',type=Path,required=True)
ap.add_argument('--later-crops-script',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();OUT=a.out;OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('lc',a.later_crops_script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
paths={'clean':a.clean,'e2':a.e2}
rows={k:[json.loads(l) for l in p.read_text().splitlines() if l.strip()] for k,p in paths.items()}
result={'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},'descriptive_not_counterfactual':True}
for cond,rs in rows.items():
 groups=collections.defaultdict(list)
 for r in rs:groups[r['uid']].append(r)
 result[cond]={'rows':len(rs),'uids':len(groups),'samples_per_uid':dict(collections.Counter(map(len,groups.values()))),'n_crops_hist':dict(sorted(collections.Counter(r['n_crops'] for r in rs).items())),'ok_vs_strict_disagree':sum(r['ok']!=r['ok_train_rule'] for r in rs),'strict_accuracy':statistics.mean(r['ok_train_rule'] for r in rs),'any_protocol_error_trajectories':sum(any(t['check_errors'] for t in r['turns']) for r in rs),'groups_by_success_count':dict(sorted(collections.Counter(sum(r['ok'] for r in g) for g in groups.values()).items()))}
 bysuccess={}
 for s in range(5):
  gs=[g for g in groups.values() if sum(r['ok'] for r in g)==s]
  bysuccess[s]={'uids':len(gs),'later_crops':sum(max(0,r['n_crops']-1) for g in gs for r in g),'long_trajectories_ge7':sum(r['n_crops']>=7 for g in gs for r in g)}
 result[cond]['by_question_success']=bysuccess
 longrs=[r for r in rs if r['n_crops']>=7]
 result[cond]['long_has_successful_sibling']=sum(any(t['ok'] for t in groups[r['uid']] if t['sample_idx']!=r['sample_idx']) for r in longrs)
 result[cond]['long_wrong_has_short_correct_sibling']=sum(not r['ok'] and any(t['ok'] and t['n_crops']<r['n_crops'] for t in groups[r['uid']]) for r in longrs)
idx={k:{(r['uid'],r['sample_idx']):r for r in rs} for k,rs in rows.items()}
pairs=[(a,idx['e2'][key]) for key,a in idx['clean'].items()]
result['paired']={'first_generated_turn_text_identical':sum(a['turns'][0]['text']==b['turns'][0]['text'] for a,b in pairs),'outcomes_clean_e2':dict(collections.Counter(f'{a["ok"]}->{b["ok"]}' for a,b in pairs)),'e2_perturb_ok':dict(collections.Counter(str((b.get('perturb') or {}).get('ok')) for a,b in pairs))}
def signature(r):
 g=[(i,t) for i,t in enumerate(r['turns']) if m.parsed(t).get('kind')=='grounding' and m.box(t)]
 geo=[]
 for j,(i,t) in enumerate(g):
  if j:geo.append(m.geometry(m.parsed(t),m.box(t),[m.box(x) for _,x in g[:j]],r['image_wh'][0]*r['image_wh'][1]))
 return geo
# Retrospective mechanism cases only; uses full trajectory for strata; not deployment/causal sample.
picks=[];used=set()
for category,pred in [('long',lambda r:r['n_crops']>=7),('repeat_or_full',lambda r:any(s in signature(r) for s in ['repeat_same_region','full_image'])),('confirm_then_answer',lambda r:r['n_crops']>=2 and any(i>=1 and ('confirm' in t['text'].lower() or 'verif' in t['text'].lower()) and m.parsed(t).get('kind')=='grounding' and i+1<len(r['turns']) and m.parsed(r['turns'][i+1]).get('kind')=='answer' for i,t in enumerate(r['turns']))),('nested_zoom',lambda r:'zoom_into_crop' in signature(r))]:
 candidates=sorted([r for r in rows['clean'] if pred(r)],key=lambda r:hashlib.sha256(f'20261023|{r["uid"]}|{r["sample_idx"]}'.encode()).hexdigest())
 for r in candidates:
  if r['uid'] in used:continue
  used.add(r['uid']);picks.append({'id':f'c{len(picks):02d}','stratum':category,'condition':'clean','uid':r['uid'],'sample_idx':r['sample_idx'],'trajectory':r})
  if sum(x['stratum']==category for x in picks)==2:break
for s in picks[:2]:
 r=idx['e2'][(s['uid'],s['sample_idx'])];picks.append({'id':f'e{len(picks):02d}','stratum':'paired_e2_long_case','condition':'e2','uid':r['uid'],'sample_idx':r['sample_idx'],'trajectory':r})
(OUT/'statistics.json').write_text(json.dumps(result,indent=2))
(OUT/'selected.json').write_text(json.dumps(picks,indent=2))
print(json.dumps(result,indent=2))
print('SELECTED',[(p['id'],p['stratum'],p['uid'],p['sample_idx'],p['trajectory']['n_crops'],p['trajectory']['ok']) for p in picks])
