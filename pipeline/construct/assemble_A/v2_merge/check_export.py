"""CPU-only validation of unchanged visible exports, independent of visual labels."""
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('${ICML_DIR}')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'obs/src'))
from export_protocol_check import check_record, HOLD
from export_selection_4k import image_paths


def main():
    dest = OUT/'cpu_candidate_check'
    dest.mkdir(exist_ok=True)
    candidates = [json.loads(s) for s in (OUT/'MANIFEST.jsonl').read_text().splitlines()]
    dataset, checks = [], []
    for s in candidates:
        rp = Path(s['result_path'])
        raw = rp.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == s['source_verified_sha256']
        ro = json.loads(raw)['rollout']
        rec = ro['export']
        images, missing = image_paths(rec, ro, rp)
        violations = sorted(set(check_record(rec)))
        hard = sorted(set(violations)&HOLD)
        packet = json.loads((OUT/'packets'/f"{s['audit_id']}.json").read_text())
        view_paths = [v['path'] for v in packet['views']]
        errors = []
        if hard: errors.append('protocol_hard_failure')
        if missing: errors.append('missing_image')
        if images != view_paths: errors.append('export_view_order_mismatch')
        if s['source_split'] != 'train': errors.append('not_train_split')
        if len(s['canonical_groups']) != 1: errors.append('not_single_original_group')
        if ro.get('termination') != 'answer': errors.append('not_completed_answer')
        assistants = [x['value'] for x in rec['conversations'] if x['from']=='gpt']
        if len(assistants) != len(packet['steps']): errors.append('assistant_turn_count')
        if len(images) != s['crops']+1: errors.append('image_crop_count')
        # The export is allowed to rename the visible caption tag, but must not
        # introduce different assertions, actions, or synthetic path splices.
        for a, step in zip(assistants, packet['steps']):
            parsed = step['parsed']['text'].replace('<caption>','<think>').replace('</caption>','</think>')
            if a.strip() != parsed.strip(): errors.append('visible_turn_differs_from_source')
        checks.append({'audit_id':s['audit_id'], 'idx':len(dataset),
                       'errors':sorted(set(errors)), 'protocol_findings':violations,
                       'source_sha256':s['source_verified_sha256']})
        dataset.append({'conversations':rec['conversations'], 'images':images})
    name = 'caption_final_candidates'
    (dest/f'{name}.json').write_text(json.dumps(dataset,ensure_ascii=False))
    (dest/'dataset_info.json').write_text(json.dumps({name:{
        'file_name':f'{name}.json','formatting':'sharegpt',
        'columns':{'messages':'conversations','images':'images'},
        'tags':{'role_tag':'from','content_tag':'value','user_tag':'human',
                'assistant_tag':'gpt','system_tag':'system'}}},indent=2))
    (OUT/'EXPORT_CHECKS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in checks))
    stats={'checked':len(checks),'passed':sum(not x['errors'] for x in checks),
           'error_counts':dict(collections.Counter(e for x in checks for e in x['errors'])),
           'status':'technical_candidate_check_not_visual_acceptance'}
    (OUT/'EXPORT_CHECK_SUMMARY.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats,ensure_ascii=False))


if __name__=='__main__':main()
