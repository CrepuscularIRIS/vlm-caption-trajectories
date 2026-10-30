"""Export the completed v53/E1 audit; run with the private workspace as argv[1].

Reuses the existing letter scorer and text redaction. No GPU/model calls.
Does not alter the earlier 23-condition export or its historical score tables.
"""
import csv
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import export_e1_rollouts as ex
from full_question import full_question


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


def clean(uid, value):
    if isinstance(value, str):
        return ex.unquote(uid, ex.redact(value))
    if isinstance(value, list):
        return [clean(uid, x) for x in value]
    if isinstance(value, dict):
        return {k: clean(uid, v) for k, v in value.items()}
    return value


def check(x):
    for s in ex.strings(x):
        for key, rx in ex.HARD_PATTERNS.items():
            assert not rx.search(s), key
        for rx in ex.STEMS.values():
            assert not rx.search(s), 'question_stem'


def main():
    w = Path(sys.argv[1]); run = w / 'obs/runs/method_v53_20261029'
    src = run / 'joint_review'; repo = ex.REPO
    logs = repo / 'logs/joint_review_20261029'; data = repo / 'data/joint_review_20261029'
    logs.mkdir(exist_ok=True); data.mkdir(exist_ok=True)
    for filename in ('val.parquet', 'vstar_holdout.parquet'):
        for r in pq.read_table(w / 'obs/runs/baselines_v52_20261024/pool' / filename).to_pylist():
            stem = full_question(r['env_head']).split('\n')[0].strip()
            if len(stem) >= 40:
                ex.STEMS[r['uid']] = re.compile(r'\s+'.join(map(re.escape, stem.split())))
    receipts = {}
    summary = json.loads((src / 'external/summary_eight.json').read_text())
    for method, tag in [('rloo', 'control'), ('rloo_decision', 'decision')]:
        raw = run / f'eval_final32/traj_v53_{tag}_s32.jsonl'
        cid = f'V53-{tag}-32'
        rows = [ex.rollout(cid, json.loads(s)) for s in raw.read_text().splitlines()]
        assert len(rows) == len({(r['item_id'], r['sample_idx']) for r in rows}) == 2208
        assert len({r['item_id'] for r in rows}) == 552
        assert round(sum(r['correct'] for r in rows)/2208, 4) == summary[method]['acc']
        check(rows)
        dest = ex.OUT / f'{cid}.jsonl.gz'
        with dest.open('wb') as f, gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0) as gz:
            gz.write(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode())
        receipts[cid] = dict(raw_sha256=sha(raw), public_sha256=sha(dest), n=2208,
                             redactions=dict(ex.REDACTED))
        ex.REDACTED.clear()
    selected = {(r['uid'], r['sample_idx'], r['method']): r for r in
                map(json.loads, (src / 'audit_records.jsonl').read_text().splitlines())}
    audit = []
    progress = json.loads((src / 'AUDIT_PROGRESS.json').read_text())
    assert progress['reviewed'] == 72 and progress['pending'] == 0
    for name in progress['findings_files']:
        for r in map(json.loads, (src / name).read_text().splitlines()):
            raw = selected[(r['uid'], r['sample_idx'], r['method'])]
            a = dict(item_id=r['uid'], sample_idx=r['sample_idx'], method=r['method'],
                     checkpoint_step=r['checkpoint_step'], gold_letter=r['gold'],
                     final_answer=r['prediction'], correct=raw['ok'],
                     original_sha256=r['original_sha256'], boxes_abs=raw['boxes_abs'],
                     image_wh=raw['image_wh'], observation=r['observation_zh'],
                     evidence_step=r['evidence_step'], evidence_quote=r['evidence_quote'],
                     protocol_errors=r['protocol_errors'], limits=r['limitations'],
                     visual_audit=r['status'])
            for k in ['first_verifiable_issue_step', 'first_issue_quote', 'categories']:
                if k in r:
                    a[k] = r[k]
            audit.append(clean(r['uid'], a))
    external = {(r['uid'], r['sample_idx'], r['model']): r for r in
                map(json.loads, (src / 'external/audit_records.jsonl').read_text().splitlines())}
    for r in map(json.loads, (src / 'external/audit_findings.jsonl').read_text().splitlines()):
        raw = external[(r['uid'], r['sample_idx'], r['model'])]
        audit.append(clean(r['uid'], dict(item_id=r['uid'], sample_idx=r['sample_idx'],
            method=r['model'], checkpoint=r['checkpoint'], gold_letter=r['gt'], correct=r['correct'],
            final_answer=ex.extract(raw['final_text'])[0], boxes_abs=raw['boxes_abs'], image_wh=raw['image_wh'],
            original_sha256=r['views']['original_sha256'], evidence_step=r['evidence_round']+1,
            evidence_quote=r['exact_quote'], observation=r['visual_observation'], limits=r['limits'],
            visual_audit=r['visual_audit'], views=[{k:v for k,v in view.items() if k!='path'}
                                                for view in r['views']['views']])))
    assert len(audit) == len({(r['item_id'], r['sample_idx'], r['method']) for r in audit}) == 96
    check(audit)
    (data / 'audit_findings.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in audit))
    dump(logs / 'summary_eight.json', summary)
    dump(logs / 'training_summary.json', json.loads((run / 'training_summary.json').read_text()))
    strata = []
    for p in [src / 'strata.csv', src / 'external/strata.csv']:
        strata.extend(csv.DictReader(p.open()))
    with (logs / 'strata.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in strata for k in r)), lineterminator='\n')
        writer.writeheader(); writer.writerows(strata)
    curves = json.loads((w / 'obs/runs/k8_20261028/curves_k8.json').read_text())
    points = {m: {s: v['val/accuracy_reward'] for s,v in curves['k8_'+m].items()
                  if 'val/accuracy_reward' in v} for m in ['grpo','dapo','gspo','gigpo']}
    for m, v in json.loads((run / 'training_summary.json').read_text()).items():
        points['rloo_'+m] = {s:r['accuracy_reward'] for s,r in v['validations'].items()}
    dump(logs / 'validation_points.json', points)
    for stem in ['eight_method_comparison','source_accuracy','crop_stratified_accuracy','six_training_curves']:
        for ext in ['png','pdf']:
            shutil.copyfile(src / 'external' / f'{stem}.{ext}', logs / f'{stem}.{ext}')
    sources = [src / f for f in progress['findings_files']] + [src / 'external/audit_findings.jsonl',
               run / 'training_summary.json', src / 'external/summary_eight.json']
    provenance = json.loads((src / 'provenance.json').read_text())
    dump(logs / 'PROVENANCE.json', dict(rollouts=receipts, audit_records=96,
         score_input_hashes={k: v['sha256'] for k,v in provenance['files'].items()},
         source_hashes={p.name:sha(p) for p in sources},
         scorer_sha256=sha(Path(__file__).parent/'summarize.py'),
         audit_export_sha256=sha(data/'audit_findings.jsonl'),
         audit_redactions=dict(ex.REDACTED),
         scope='Directed single-reviewer visual audit with unresolved items; no causal attribution'))
    print('Export checked: 4416 new rollouts, 96 unique audit records; original outputs unchanged.')


if __name__ == '__main__':
    main()
