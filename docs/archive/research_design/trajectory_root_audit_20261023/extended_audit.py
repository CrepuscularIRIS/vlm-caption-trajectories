"""Descriptive existing-log analysis; question-cluster bootstrap, no model calls."""
import argparse,collections,json,random,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--clean',type=Path,required=True);ap.add_argument('--e2',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
rows={k:[json.loads(l) for l in p.read_text().splitlines() if l.strip()] for k,p in [('clean',a.clean),('e2',a.e2)]}
out={};gs={}
for cond,rs in rows.items():
 groups=collections.defaultdict(list)
 for r in rs:groups[r['uid']].append(r)
 gs[cond]=groups;by={}
 for source in sorted({r['source'] for r in rs}):
  rr=[r for r in rs if r['source']==source];uids={r['uid'] for r in rr}
  by[source]={'uids':len(uids),'all_4_wrong_uids':sum(not any(x['ok'] for x in groups[u]) for u in uids),'long_ge7':sum(r['n_crops']>=7 for r in rr),'later_crops':sum(max(0,r['n_crops']-1) for r in rr)}
 counts=[]
 for r in rs:
  seen=set();n=0
  for t in r['turns']:
   if 'box_abs' not in t:continue
   b=tuple(t['box_abs']);n+=b in seen;seen.add(b)
  counts.append(n)
 out[cond]={'by_source':by,'repeated_exact_executed_boxes':sum(counts),'rows_with_repeated_exact_box':sum(n>0 for n in counts)}
uids=sorted(gs['clean']);assert set(uids)==set(gs['e2'])
d=[statistics.mean(r['ok'] for r in gs['e2'][u])-statistics.mean(r['ok'] for r in gs['clean'][u]) for u in uids]
rng=random.Random(20261023);b=sorted(statistics.mean(rng.choices(d,k=len(d))) for _ in range(5000))
# Explicit nearest-index percentile; slight library percentile convention differences possible.
out['paired_accuracy_question_cluster']={'mean':statistics.mean(d),'percentile_ci95':[b[125],b[4874]],'n_uid_clusters':len(d),'image_family_clusters_not_available':True,'bootstrap_seed':20261023,'replicates':5000}
a.out.write_text(json.dumps(out,indent=2)+'\n')
