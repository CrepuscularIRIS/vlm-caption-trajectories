#!/usr/bin/env python3
"""Verify the frozen October 6 progress release and render its text casebook.

Standard library only; no model calls, credentials, network, or GPU.
Passing checks establish artifact consistency, not visual truth or SFT acceptance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return json.loads((ROOT / rel).read_text())


def casebook(cases):
    out = ['# 2026-10-06 同题四模型校准：6 题、24 条真实路径', '',
        '这些是定向校准材料，不是随机质量样本，也不是已验收 SFT。模型原判与后续复核意见分别保留；没有填写人工 KEEP。', '',
        '[进度与模型分析](13_progress_20261006.md) · [历史案例册](CASEBOOK_20261004.md) · '
        '[图片发布范围](11_publication_and_reproduction_20261004.md)', '',
        '仅复用已公开的 WorldBench 视图；MME 原图/crop 不公开，须在合法持有数据的本地环境按路径和哈希查看。'
        '公开路径均相对于原研究工作区，不是本仓库内一定存在的文件。', '',
        '|案例|要核对的行为|', '|---|---|']
    out += [f"|[{c['case_id']} · {c['item_id']}](#{c['case_id'].lower()})|{c['title']}|" for c in cases]
    for c in cases:
        out += ['',f'<a id="{c["case_id"].lower()}"></a>','',f"## {c['case_id']} · {c['title']}",'',
            '**任务转述：** '+c['task_paraphrase_zh'],'','**复核意见：** '+c['editorial_observation_zh'],'',
            '|生成器|终答 / GT|步数 / crop|模型原裁决|显式分歧 / 待核标记|', '|---|---|---|---|---|']
        for t in c['trajectories']:
            out.append(f"|{t['label']}|{t['answer']} / {t['gold']}|{len(t['steps'])} / {t['crops']}|"
                f"{t['model_adjudication']['verdict']}|{t['reviewers_split']} / {t['ambiguous']}|")
        out += ['','**原裁决不是最终入池许可。** P03/P04/P06 的复核说明优先用于解释争议；尤其 P06 GLM 同时被标 clean 与 ambiguous，不能当 clean 导出。']
        shown=set()
        for t in c['trajectories']:
            out += ['',f"### {t['label']} · {t['model']}",'']
            views={v['view_id']:v for v in t['views']}
            def show(vid):
                v=views[vid];asset=v.get('public_asset')
                if asset and asset not in shown:
                    shown.add(asset)
                    return ['',f"![{c['case_id']} {t['label']} {vid}](../{asset})",'']
                if asset:return ['',f"视图 `{vid}`：[已公开图像](../{asset})。",'']
                return ['',f"视图 `{vid}`：本地查看，SHA-256 `{v['sha256']}`。",'']
            out += show('original_image')
            for step in t['steps']:
                out += [f"**Step {step['step']} · {step['behavior']}**；此前可用：`{', '.join(step['available_before'])}`。",'',
                    '```text',step['visible_output'],'```']
                if step['observation']:out += show(step['observation'])
            out += ['', '**模型原判理由：** '+t['model_adjudication']['reason'],'',
                '<details><summary>来源与时序元数据</summary>','',f"- 源结果：`{t['source_path']}`",
                f"- 源哈希：`{t['source_sha256']}`",'- 人工终裁：未填写。','','</details>']
        out += ['',f"[完整可见路径 JSON](../cases/20261006/{c['case_id']}_{c['item_id']}.json)"]
    return '\n'.join(out)+'\n'


def verify():
    s=read('data/progress_20261006.json')
    rows=[json.loads(l) for l in (ROOT/'data/consensus_audit_rows_20261006.jsonl').read_text().splitlines()]
    initial=[r for r in rows if r['phase'].startswith('initial')]
    swap=[r for r in rows if r['phase']=='swap1']
    assert len(initial)==s['initial']['tasks']==1984
    assert len({r['item_id'] for r in initial})==len(initial)
    assert Counter(r['pool'] for r in initial)==s['initial']['pools']
    assert Counter(r['decision'] for r in initial if r['phase']=='initial_new')==s['initial']['decisions']
    assert sum(r['phase']=='initial_reused' for r in initial)==s['initial']['reused']==260
    assert len(swap)==s['swap1']['completed'] and len({r['item_id'] for r in swap})==len(swap)
    assert Counter(r['decision'] for r in swap)==s['swap1']['decisions']
    initial_by_id={r['item_id']:r for r in initial}
    assert all(initial_by_id[r['item_id']]['pool']=='rejected' for r in swap)
    assert all(initial_by_id[r['item_id']]['model']!=r['model'] for r in swap)
    clean=[r for r in initial if r['pool']=='clean_candidate']
    assert len(clean)==len({r['group_id'] for r in clean})==1352
    for m,v in s['initial']['by_model'].items():
        rs=[r for r in initial if r['model']==m];cs=[r for r in rs if r['pool']=='clean_candidate']
        assert len(rs)==v['selected'] and len(cs)==v['auto_candidates']
        assert Counter(r['pool'] for r in rs)==v['pools']
        assert sum(r['crops']==0 for r in cs)==v['candidate_direct']
        assert sum(r['crops']>0 for r in cs)==v['candidate_tool']
    assert sum(x['records'] for x in s['inventory']['per_model'].values())==84880
    assert sum(s['inventory']['per_model'][m]['records'] for m in s['initial']['by_model'])==82816
    spots=read('data/spotcheck132_20261006.json')
    assert len(spots)==s['spotcheck']['total']==132
    assert Counter(x['pool'] for x in spots.values())==s['spotcheck']['pools']
    assert Counter(x['verdict'] for x in spots.values())==s['spotcheck']['verdicts']
    assert all(s['accepted_sft'][k] is None for k in ['T','Q','G'])
    unreviewed = [r for r in clean if r['fallback_unreviewed']]
    gap = s['known_gaps']['fallback_unreviewed']
    assert len(unreviewed) == gap['initial_auto_candidates'] == 51
    assert Counter(r['model'] for r in unreviewed) == gap['initial_by_model']
    assert Counter(r['phase'] for r in unreviewed) == gap['initial_by_phase']
    assert sum(r['fallback_unreviewed'] and r['decision'] == 'AUTO_PASS_PENDING_MANUAL'
               for r in swap) == gap['swap_auto_candidates'] == 11
    sources=read(s['provenance_file'])['source_files']
    for r in rows:
        assert r['human_verdict'] is None
        for kind in ['result','audit']:
            p=r[kind+'_path'];assert not Path(p).is_absolute() and '..' not in Path(p).parts
            assert sources[p]==r[kind+'_sha256']
    cases=[json.loads(p.read_text()) for p in sorted((ROOT/'cases/20261006').glob('*.json'))]
    assert len(cases)==6 and sum(len(c['trajectories']) for c in cases)==24
    for c in cases:
        assert len({t['family'] for t in c['trajectories']})==3
        for t in c['trajectories']:
            assert t['manual_verdict'] is None and sources[t['source_path']]==t['source_sha256']
            vs={v['view_id']:v for v in t['views']};available=['original_image']
            assert len(vs)==t['crops']+1
            for step in t['steps']:
                assert step['available_before']==available and set(step['cited'])<=set(available)
                if step['observation']:
                    v=vs[step['observation']]
                    assert v['parent'] in available and step['action']['source']==v['parent']
                    available.append(step['observation'])
            assert set(available)==set(vs)
            for v in vs.values():
                if v['public_asset']:
                    assert c['item_id'].startswith('worldbench_')
                    assert hashlib.sha256((ROOT/v['public_asset']).read_bytes()).hexdigest()==v['sha256']
    return s,cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--write-casebook',action='store_true');args=p.parse_args()
    s,cases=verify();path=ROOT/'docs/CASEBOOK_20261006.md';text=casebook(cases)
    if args.write_casebook:path.write_text(text)
    else:assert path.read_text()==text,'Casebook does not match frozen case JSON'
    print(json.dumps({'consistency':'PASS','captured_at':s['captured_at'],'initial_tasks':1984,
        'auto_candidates':1352,'swap_reviewed':s['swap1']['completed'],'human_pending':132,
        'initial_candidates_fallback_unreviewed':51,'swap_candidates_fallback_unreviewed':11,
        'cases':6,'paths':24},indent=2))
    print('Consistency is not visual truth, independent judge agreement, or training acceptance.')


if __name__=='__main__':main()
