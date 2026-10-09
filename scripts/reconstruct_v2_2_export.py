#!/usr/bin/env python3
"""Reconstruct the exact selected ShareGPT records from legally held local assets.

No downloads, image redistribution, model calls, or training. Outputs must be
outside this public checkout. The published generated text is not a source-data
license and does not grant permission to redistribute benchmark images.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PUB=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--workspace-root',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--increment-source-index',type=Path,required=True,help='Private R2 source index; never uploaded.');args=a.parse_args()
    root=args.workspace_root.resolve();out=args.out.resolve()
    if out.is_relative_to(PUB):raise SystemExit('Output must be outside the public checkout.')
    if out.exists() and any(out.iterdir()):raise SystemExit('Use a new empty output directory.')
    subprocess.run([sys.executable,str(PUB/'scripts/verify_v2_2_release.py')],check=True)
    traces=[json.loads(s) for name in ('v2_merged_keep_trajectories_20261008.jsonl','v2_1_increment_keep_20261008.jsonl','v2_2_increment_keep_20261009.jsonl') for s in (PUB/'data'/name).read_text().splitlines()]
    private_index={r['result_sha256']:Path(r['result']).resolve() for r in json.loads(args.increment_source_index.read_text())}
    dataset=[]
    for r in traces:
        is_increment=r.get('text_format')=='visible_caption_raw'
        p=private_index[r['result_sha256']] if is_increment else (root/r['source_result']).resolve()
        assert p.is_relative_to(root) and digest(p)==r['result_sha256'],r['audit_id']
        ro=json.loads(p.read_text())['rollout'];rec=ro['export']
        expected=r['assistant_turns']
        if is_increment:
            assert [s['raw_text'] for s in ro['steps']]==expected
            expected=[t.replace('<caption>','<think>').replace('</caption>','</think>') for t in expected]
        assert [c['value'] for c in rec['conversations'] if c['from']=='gpt']==expected
        images=[]
        assert len(r['views'])==len(rec['images'])
        for vi,v in enumerate(r['views']):
            image=(root/rec['images'][vi]).resolve() if is_increment else (root/v['source_path']).resolve()
            assert image.is_relative_to(root) and digest(image)==v['sha256'],(r['audit_id'],v['view_id'])
            images.append(str(image))
        source_images=[str((root/p).resolve()) for p in rec['images']]
        assert source_images==images,(r['audit_id'],'source image order differs')
        dataset.append({'conversations':rec['conversations'],'images':images})
    out.mkdir(parents=True,exist_ok=True);name='caption_sft_v2_2_keep'
    (out/f'{name}.json').write_text(json.dumps(dataset,ensure_ascii=False))
    (out/'dataset_info.json').write_text(json.dumps({name:{'file_name':f'{name}.json','formatting':'sharegpt',
        'columns':{'messages':'conversations','images':'images'},'tags':{'role_tag':'from','content_tag':'value',
        'user_tag':'human','assistant_tag':'gpt','system_tag':'system'}}},indent=2)+'\n')
    print(json.dumps({'reconstructed_paths':len(dataset),'output':str(out),'no_training':True},indent=2))

if __name__=='__main__':main()
