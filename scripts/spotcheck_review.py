#!/usr/bin/env python3
"""Render and validate the frozen 75-case review packet, without model calls."""
import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return json.loads((ROOT / rel).read_text())


def require(ok, why):
    if not ok:
        raise ValueError(why)


def markdown(spot):
    n=len(spot)
    final=n==75
    out = [f'# {n} 条分层抽检：完整可见步骤', '',
        f'**已抽取并生成案例册；{n} 条人工终裁仍为 PENDING。** '
        + ('来自全审完的 1,317 候选，原 69 条全部保留，新增 6 条。' if final else '这是原池 1,241 候选的历史 69 条；新版 75 条已全部保留它们，见 [当前抽检册](SPOTCHECK75_20261004.md)。') +
        '本文是文本审阅入口，不能替代原图核验。', '',
        '[新增审核与重做分析](12_cc_spotcheck_update_20261004.md) · '
        f'[结构化样本与视图哈希](../data/spotcheck{n}_20261004.json)', '',
        '第三方题目原文和新图像不在此公开页面分发；保留模型实际可见 caption、动作和时序。'
        '完整问题、原图与 crop 在本地 `obs/runs/audit_luna_v1_20261003/SPOTCHECK_CASEBOOK.html` 查看。'
        '旧页面的“Gemini read”是固定文案，真实 reader 须以 sidecar 为准。', '',
        '|样本|题目|分层|步数 / crop|人工状态|回退漏审|', '|---|---|---|---|---|---|']
    for c in spot:
        out.append(f"|[{c['sample_id']}](#{c['sample_id'].lower()})|{c['item_id']}|{c['stratum']}|{len(c['steps'])} / {c['crops']}|{c['manual_verdict']}|{'是' if c['fallback_unreviewed'] else '否'}|")
    for c in spot:
        out += ['',f'<a id="{c["sample_id"].lower()}"></a>','',
                f"## {c['sample_id']} · {c['item_id']}",'',
                f"`{c['stratum']}` · 模型 `{c['model']}` · 终答 / 原金标 `{c['answer']} / {c['gold']}` · 人工 `{c['manual_verdict']}`",'',
                '<details><summary>展开完整可见步骤与来源</summary>','']
        for s in c['steps']:
            out += [f"**Step {s['step']}** · 此前已收到：`{'`, `'.join(s['available_before'])}`。",'',
                    '```text',s['visible_output'],'```','']
            if s['observation']:
                v=next(v for v in c['views'] if v['view_id']==s['observation'])
                out += [f"动作返回 `{s['observation']}`；保存视图 SHA-256 `{v['sha256']}`。",'']
        out += [f"原结果：`{c['result_path']}`",'',f"原结果 SHA-256：`{c['result_sha256']}`",'',
                '建议人工记录：KEEP / WEAK / BAD / UNCERTAIN、问题步骤、像素依据，以及是否存在跨视图绑定或无据断言。'
                '本次没有替人填写意见，也没有将整个抽检集作为训练正例导出。','', '</details>']
    return '\n'.join(out)+'\n'


def verify():
    s=read('data/cc_update_20261004.json')
    rows=[json.loads(x) for x in (ROOT/'data/audit_rows_cc_20261004.jsonl').read_text().splitlines()]
    old=[json.loads(x) for x in (ROOT/'data/production_audit_rows_20261003.jsonl').read_text().splitlines()]
    oldby={r['key']:r for r in old}
    require(len(rows)==s['expected_manifest']==2520,'Manifest count')
    require({r['key'] for r in rows}==set(oldby),'Manifest identity drift')
    require(Counter(r['pool'] for r in rows)==s['pools'],'Current pools')
    require(Counter(r['previous_pool']+' -> '+r['pool'] for r in rows)==s['transitions'],'Transitions')
    require(all(r['previous_pool']==oldby[r['key']]['pool'] for r in rows),'Previous pools')
    require(sum(r['pool']!='pending' for r in rows)==s['completed'],'Completion count')
    require(s['completed']==2520 and s['missing_current_sidecars']==0,'Final audit must be complete')
    require(s['on_disk_pool_summary']['pools']==s['pools'],'Production pool summary differs')
    require(s['human_review_queue']==s['pools']['disputed']+s['pools']['review_format']==287,'Human queue')
    cc=[r for r in rows if r['process_route']=='cc_proxy']
    require(len(cc)==s['cc_completed']==141,'CC complete')
    require(Counter(r['decision'] for r in cc)==s['cc_decisions'],'CC decisions')
    clean=[r for r in rows if r['pool']=='clean_candidate']
    require(len({r['group_id'] for r in clean})==s['clean_groups'],'Clean groups')
    require(sum(r['fallback_unreviewed'] for r in clean)==s['clean_fallback_unreviewed']==81,'Fallback remains pending')
    receipts=read('data/cc_receipts_20261004.json')
    require(len(receipts)==s['cc_receipts']==273,'CC receipt count')
    require(s['cc_budget_dispatches']==len(receipts)+len(s['cc_archived_smoke'])==275,'Smoke failures omitted from dispatch count')
    require(len({r['request_sha256'] for r in receipts})==len(receipts),'Unique receipt IDs')
    for r in receipts:
        require(r['valid'] and r['independently_reparsed_valid'] and r['parsed_fields_match'],'Receipt validation')
        require(r['exit']==0 and r['all_images_opened'] and not r['disallowed_tool_count'] and not r['unexpected_path_count'],'Receipt safety/transport')
        require(r['image_count']==r['staged_image_count'],'Image staging count')
    require(all(r['structure_receipt'] in {v['request_sha256'] for v in receipts} for r in cc),'CC structure links')
    spot=read('data/spotcheck75_20261004.json')
    require(len(spot)==75 and len({c['key'] for c in spot})==75,'Spotcheck identity')
    require(Counter(c['manual_verdict'] for c in spot)=={'PENDING':75},'No invented manual verdict')
    require(sum(len(c['steps']) for c in spot)==s['spotcheck']['total_steps']==132,'Visible step count')
    for stratum in s['spotcheck']['strata']:
        k,n=stratum['stratum'],stratum['base_candidates']
        expected=min(n,max(2,math.ceil(.05*n)))
        eligible=[r for r in rows if r['pool']=='clean_candidate' and r['tag']+'/'+r['material']==k]
        require(len(eligible)==n,'Original stratum size')
        selected=sorted(eligible,key=lambda r:hashlib.sha256(('spot'+r['key']).encode()).hexdigest())[:expected]
        require({r['key'] for r in selected}=={c['key'] for c in spot if c['stratum']==k},'Deterministic sample drift')
    original=read('data/spotcheck69_20261004.json')
    require({c['key'] for c in original}<={c['key'] for c in spot},'Original 69 lost')
    require(len(spot)-len(original)==6,'Incremental sample size')
    for c in spot:
        require(oldby[c['key']]['result_sha256']==c['result_sha256'],'Original sample result hash')
        views={v['view_id']:v for v in c['views']};available=['original_image']
        for i,step in enumerate(c['steps'],1):
            require(step['step']==i and step['available_before']==available,'Spotcheck prefix')
            if step['observation']:
                v=views[step['observation']]
                require(v['parent'] in available and v['view_id'] not in available,'View ancestry')
                available.append(v['view_id'])
        require(len(available)==len(views)==c['crops']+1,'View count')
    triage=read('data/rejection_triage_20261004.json')
    require(Counter(t['suggested_action'] for t in triage)==s['triage'],'Triage counts')
    require(Counter(t['suggested_action'] for t in triage if t['pool']=='rejected')==s['rejected_triage'],'Rejected triage')
    for t in triage:
        require(not t['generated_now'] and not t['semantic_truth_confirmed'],'Proposal must stay a proposal')
        if t['suggested_action']=='review_existing_alternative_first':
            require(any(a['score']==1 and a['termination']=='answer' for a in t['alternatives']),'Missing existing alternative')
    require((ROOT/'docs/SPOTCHECK75_20261004.md').read_text()==markdown(spot),'Spotcheck Markdown mismatch')
    require((ROOT/'docs/SPOTCHECK69_20261004.md').read_text()==markdown(original),'Original spotcheck Markdown mismatch')
    for rel,record in read('data/cc_code_provenance_20261004.json')['published_code'].items():
        require(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==record['published_sha256'],'CC code snapshot hash')
    for r in rows:
        for k in ('result_path','audit_path'):
            require(not Path(r[k]).is_absolute() and '..' not in Path(r[k]).parts,'Unsafe path')
    print(json.dumps(dict(consistency='PASS',completed=s['completed'],automatic_candidates=len(clean),
        cc_completed=len(cc),cc_receipts=len(receipts),spotcheck=75,manual_pending=75,rejected=916),indent=2))
    print('PASS verifies frozen evidence consistency, not visual truth or human acceptance.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write-markdown',action='store_true')
    ap.add_argument('--verify',action='store_true')
    ap.add_argument('--workspace-root',type=Path)
    ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if a.write_markdown:
        (ROOT/'docs/SPOTCHECK69_20261004.md').write_text(markdown(read('data/spotcheck69_20261004.json')))
        (ROOT/'docs/SPOTCHECK75_20261004.md').write_text(markdown(read('data/spotcheck75_20261004.json')))
    if a.verify:
        verify()
    if a.workspace_root and a.output:
        s=read('data/cc_update_20261004.json')['spotcheck']
        workspace=a.workspace_root.resolve();source=(workspace/s['source_html']).resolve();dest=a.output.resolve()
        require(source.is_relative_to(workspace),'Source escapes workspace')
        require(not dest.is_relative_to(ROOT) and dest!=source,'Private output must be separate from repository/source')
        require(hashlib.sha256(source.read_bytes()).hexdigest()==s['source_html_sha256'],'Source casebook version changed')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        print('Copied unchanged local visual book; do not redistribute third-party images: '+str(dest))
    elif a.workspace_root or a.output:
        ap.error('--workspace-root and --output must be used together')
    elif not (a.write_markdown or a.verify):
        ap.error('Choose verification, markdown generation, or a local visual book copy')


if __name__=='__main__':
    main()
