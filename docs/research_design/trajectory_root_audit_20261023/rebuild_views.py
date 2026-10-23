"""CPU-only reconstruction of stored crop actions; no model/token generation."""
import ast,hashlib,inspect,json,sys
from pathlib import Path
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--env-dir',type=Path,required=True);ap.add_argument('--easyr1',type=Path,required=True);ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--work-dir',type=Path,required=True)
a=ap.parse_args()
sys.path.insert(0,str(a.env_dir));sys.path.insert(0,str(a.easyr1))
import pandas as pd
# Extract the already-inspected pure image function unchanged, avoiding training-only imports.
import math
from io import BytesIO
from typing import Any, Optional, Union
from PIL import Image
from PIL.Image import Image as ImageObject
_dataset_source=(a.easyr1/'verl/utils/dataset.py').read_text()
_fn=next(n for n in ast.parse(_dataset_source).body if isinstance(n,ast.FunctionDef) and n.name=='process_image')
_process_source=ast.get_source_segment(_dataset_source,_fn)
exec(compile(_process_source,'<existing_process_image>','exec'))
import caption_env as E
BASE=a.work_dir
picks=json.loads((BASE/'selected.json').read_text())
rows={r['uid']:r for name in ('val','vstar_holdout') for _,r in pd.read_parquet(a.data_dir/(name+'.parquet')).iterrows()}
manifest=[]
for c in picks:
 d=BASE/'views'/c['id'];d.mkdir(parents=True,exist_ok=True)
 row=rows[c['uid']];ex=row['extra_info'];ex=ast.literal_eval(ex) if isinstance(ex,str) else ex
 env=E.CropEnv(row['images'][0]);t=c['trajectory'];recs=[]
 assert list(env.full.size)==t['image_wh']
 def save():
  k=len(env.images)-1;im=process_image(env.images[-1],1024,589824);p=d/f'view_{k}.png';im.save(p)
  recs.append({'view':k,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'shown_wh':list(im.size),'box_abs':list(env.boxes[-1])})
 save()
 for turn in t['turns']:
  if 'box_abs' not in turn:continue
  p=turn['parsed'];p=ast.literal_eval(p) if isinstance(p,str) else dict(p)
  if env.crops==0 and c['condition']=='e2' and t.get('perturb') and 'shown_bbox' in t['perturb']:p['bbox']=t['perturb']['shown_bbox']
  result=env.step(p)
  assert result is not None and list(env.boxes[-1])==turn['box_abs'],(c['id'],turn['round'],'box mismatch')
  save()
 manifest.append({'id':c['id'],'uid':c['uid'],'sample_idx':c['sample_idx'],'condition':c['condition'],'question':ex['question'],'views':recs,'reconstruction':'stored executed boxes matched; process_image replay; not recorded production token ledger'})
(BASE/'view_manifest.json').write_text(json.dumps({'env_sha256':hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(),'process_image_source_sha256':hashlib.sha256(_process_source.encode()).hexdigest(),'cases':manifest},indent=2))
print(json.dumps({'cases':len(manifest),'views':sum(len(x['views']) for x in manifest),'all_boxes_matched':True}))
