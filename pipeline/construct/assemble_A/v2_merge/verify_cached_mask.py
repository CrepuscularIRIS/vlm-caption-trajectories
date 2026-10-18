"""Check already cached CPU preprocessing; no image processing or model weights."""
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

OUT=Path(__file__).resolve().parent
P=OUT/'cpu_candidate_check'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    caches=[]
    for path in (P/'cache').rglob('cache-*.arrow'):
        ds=Dataset.from_file(str(path))
        if 'input_ids' in ds.column_names:caches.append((path,ds))
    assert len(caches)==1
    path,ds=caches[0]
    raw=json.loads((P/'caption_final_candidates.json').read_text())
    assert len(ds)==len(raw)==5378
    tok=AutoTokenizer.from_pretrained('${MODEL_DIR}/Qwen2.5-VL-7B-Instruct',local_files_only=True)
    start=tok.convert_tokens_to_ids('<|im_start|>')
    assistant_header=tok.encode('<|im_start|>assistant\n',add_special_tokens=False)
    issues=[]
    for idx,(ex,source) in enumerate(zip(ds,raw)):
        ids,labels=ex['input_ids'],ex['labels']
        expected=''.join(c['value']+'<|im_end|>\n' for c in source['conversations'] if c['from']=='gpt')
        actual=tok.decode([v for v,l in zip(ids,labels) if l!=-100],skip_special_tokens=False)
        if actual!=expected:issues.append({'idx':idx,'error':'assistant_target_text_mismatch'})
        if any(l!=-100 and l!=v for v,l in zip(ids,labels)):
            issues.append({'idx':idx,'error':'label_token_value_mismatch'})
        starts=[i for i,v in enumerate(ids) if v==start]+[len(ids)]
        expected_mask=[False]*len(ids)
        for first,last in zip(starts,starts[1:]):
            if ids[first:first+len(assistant_header)]==assistant_header:
                expected_mask[first+len(assistant_header):last]=[True]*(last-first-len(assistant_header))
        if expected_mask!=[l!=-100 for l in labels]:
            issues.append({'idx':idx,'error':'non_assistant_or_missing_assistant_loss_position'})
    above=collections.Counter();view_count=0
    for f in (OUT/'packets').glob('*.json'):
        row=json.loads(f.read_text())
        for view in row['views']:
            view_count+=1
            w,h=view['shown_px']
            if w*h>768*768:above['original' if view['view_id']=='original_image' else 'crop']+=1
    result={'status':'PASS' if not issues else 'FAIL','samples':len(ds),'exact_mask_and_target_passed':len(ds)-len({r['idx'] for r in issues}),
      'issues':issues,'cache_sha256':digest(path),'raw_dataset_sha256':digest(P/'caption_final_candidates.json'),
      'check_script_sha256':digest(Path(__file__)),
      'runtime_versions':{n:importlib.metadata.version(n) for n in ('transformers','datasets','torch','llamafactory')},
      'processor_max_pixels':768*768,'processor_min_pixels':32*32,'train_on_prompt':False,'mask_history':False,
      'saved_views':view_count,'saved_views_above_pixel_budget':dict(above),
      'pixel_budget_note':'View dimensions exceed the preprocessing pixel budget; no claim that student-level text or details stay readable.',
      'no_GPU':True,'no_training':True,'no_image_reprocessing':True}
    (OUT/'CPU_MASK_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
