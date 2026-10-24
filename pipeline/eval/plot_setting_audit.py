"""Plot existing measured historical results, never fill missing baselines."""
from pathlib import Path
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'logs/setting_audit_20261023'
OUT = ROOT/'media/setting_audit_20261023'
OUT.mkdir(parents=True, exist_ok=True)
rows=json.loads((DATA/'existing_eval_audit.json').read_text())['runs']
lookup={r['path']:r for r in rows}
# Names refer to the actual historical run, not planned Base+GRPO baselines.
models=[
 ('A SFT','v3_behavior/traj_A_sft'),
 ('Answer SFT','v3_behavior/traj_AO_sft'),
 ('Answer SFT+RL','v3_behavior/traj_AO_rl160'),
 ('A GRPO original','v3_behavior/traj_A_rl'),
 ('A GRPO rerun','dvgrpo_20261020/traj_orig160'),
 ('B SFT','v3_behavior/traj_B_sft'),
 ('B SFT+GRPO','v3_behavior/traj_B_rl'),
 ('GRPO + KL','v3_behavior/traj_A_kl160'),
 ('Cost .01 @40','v3_behavior/traj_A_cc40'),
 ('Floor .025','v3_behavior/traj_A_fl025'),
 ('Floor .05','v3_behavior/traj_A_fl050'),
 ('Card A (adapted)','v3_behavior/traj_A_cc160'),
 ('Card B (+DS)','v3_behavior/traj_A_ccb160'),
 ('First-turn dV','dvgrpo_20261020/traj_dv160'),
 ('RLOO 0','rloo_20261021/traj_r0_160'),
 ('RLOO .01','rloo_20261021/traj_r1_160'),
 ('Four-case credit','fc_20261022/traj_fc160'),
 ('Zero credit','fcz_20261022/traj_zero160'),
]
colors=list(plt.get_cmap('tab20').colors)
fig,axes=plt.subplots(2,3,figsize=(17,10),sharey=True)
fields=[('total_logical_tokens','Total logical processed tokens / sample',1000),
        ('first','First crop / sample',1),('later','Later crops / sample',1)]
export=[]
for i,(name,path) in enumerate(models):
 for row,suffix in enumerate(['','_perturb']):
  r=lookup.get('evidence/'+path+suffix+'.jsonl')
  if r is None:continue
  assert r['complete_552x4'], r['path']
  assert abs(r['crops']-r['first']-r['later'])<1e-10
  assert abs(r['total_logical_tokens']-r['prompt_tokens']-r['gen_tokens'])<1e-8
  export.append(dict(id=i+1,label=name,eval='E1' if row==0 else 'E2',**r))
  for j,(key,label,div) in enumerate(fields):
   ax=axes[row,j];x=r[key]/div;y=100*r['acc_ok']
   ax.scatter(x,y,s=55,color=colors[i],edgecolor='black',linewidth=.4)
   # Numbered points keep identities legible without long overlapping names.
   ax.annotate(str(i+1),(x,y),xytext=(5,5),textcoords='offset points',fontsize=8,
               arrowprops=dict(arrowstyle='-',color=colors[i],lw=.6),ha='center',va='center')
for row in range(2):
 for j,(_,label,div) in enumerate(fields):
  ax=axes[row,j];ax.grid(alpha=.2);ax.set_xlabel(label+(' (thousands)' if div==1000 else ''))
  ax.set_title(('E1: natural task' if row==0 else 'E2: first-crop perturbation')+' | '+['all-token cost','first crop','later crops'][j],fontsize=11)
  ax.spines[['top','right']].set_visible(False)
  ax.set_ylim(44,70)
 axes[row,0].set_ylabel('Final-answer accuracy (%)')
handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i],label=f'{i+1}. {n}',markersize=7) for i,(n,_) in enumerate(models)]
fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,fontsize=9,bbox_to_anchor=(.5,.035))
fig.suptitle('Historical accuracy–cost comparison | same E1/E2 questions | NOT the new baseline study',fontsize=15,y=.985)
fig.text(.5,.935,'552 questions x 4 samples, T=0.7; RL @160 unless labeled. Point estimates only; no non-inferiority claim.',ha='center',fontsize=10)
fig.text(.5,.012,'Missing Base zero-shot / external models and 64-question DL are not imputed. E2 no-tool conditions are not applicable. Logical tokens include replayed history and image tokens.',ha='center',fontsize=9)
fig.subplots_adjust(top=.88,bottom=.22,hspace=.30,wspace=.14)
# Place numeric labels without moving data points. Leader lines preserve identity
# when the first-crop probabilities are close to one another.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for ax in axes.flat:
 occupied=[]
 for ann in ax.texts:
  best=None
  for radius in [8,15,23,32,45,60]:
   for dx,dy in [(radius,0),(-radius,0),(0,radius),(0,-radius),(radius,radius),(-radius,radius),(radius,-radius),(-radius,-radius)]:
    ann.set_position((dx,dy))
    box=ann.get_window_extent(renderer).expanded(1.12,1.12)
    # Annotation extent includes its arrow. Use just its text box for packing.
    from matplotlib.text import Text
    box=Text.get_window_extent(ann,renderer).expanded(1.25,1.2)
    outside=not (ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1))
    score=sum(box.overlaps(old) for old in occupied)*10000+outside*10000+dx*dx+dy*dy
    if best is None or score<best[0]:best=(score,(dx,dy),box.frozen())
  ann.set_position(best[1]);occupied.append(best[2])
for ext in ['png','svg']:fig.savefig(OUT/f'historical_tradeoff.{ext}',dpi=180)
plt.close(fig)
(DATA/'historical_tradeoff_data.json').write_text(json.dumps(export,indent=2)+'\n')

grad=json.loads((DATA/'gradient_audit.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
names={'grpo_v3_A':'Original GRPO-A','grpo_v3_A_floor_ctrl_l000':'Control rerun',
       'grpo_v3_A_floor_l025':'Floor .025','grpo_v3_A_floor_l050':'Floor .05','grpo_v3_A_cc_dsasym':'Card B'}
for r in grad:
 if r['run'] not in names:continue
 axes[0].plot([v['step'] for v in r['validation']],[100*v['accuracy'] for v in r['validation']],marker='o',label=names[r['run']],linewidth=1.4)
rr=list(csv.DictReader((ROOT/'logs/grpo/grpo_v3_A_metrics.csv').open()))
rr=[r for r in rr if r.get('actor/grad_norm')]
axes[1].plot([int(r['step']) for r in rr],[float(r['actor/grad_norm']) for r in rr],lw=1,label='Pre-clip norm: GRPO-A')
axes[1].axhline(.1,c='red',ls='--',label='Historical max_grad_norm = 0.1')
axes[1].axhline(1,c='gray',ls=':',label='Planned new threshold = 1.0 (not tested)')
axes[1].set_yscale('log');axes[1].set_ylabel('Gradient norm (log scale)')
axes[0].set_ylabel('Training-validation accuracy (%)')
for ax in axes:ax.set_xlabel('Optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Training-setting audit | validation curves are not E1; gradients are pre-clipping',fontsize=12)
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(OUT/f'training_diagnostics.{ext}',dpi=180)
print('saved plots and',len(export),'historical points')
