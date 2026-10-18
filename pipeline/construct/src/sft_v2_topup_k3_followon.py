"""Continue only the 111 never-dispatched top-up tasks with K3.

Reuses the frozen generator/transport without changing previous runs or exports.
No automatic restart, second attempt, process promotion, or other model route.
"""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import shutil
import time

import sft_v2_topup as gen

ROOT = gen.ROOT
PREVIOUS = ROOT / 'obs/runs/sft_v2_topup_20261008'
BASELINE = ROOT / 'obs/benchmarks/sft_v2_1_keep_20261008'
ORIGINAL_STATUS = gen.status


def choose_unstarted(items, completed_ids, baseline):
    selected = [dict(i, assigned_role='k3') for i in items
                if i['assigned_role'] == 'gemini' and i['item_id'] not in completed_ids]
    keys = {r['canonical_sample_key'] for r in baseline}
    groups = {g for r in baseline for g in r['canonical_groups']}
    seen_keys, seen_groups = set(), set()
    for it in selected:
        key, group = it['actual_sample_key'], it['group_id']
        if key in keys or group in groups or key in seen_keys or group in seen_groups:
            raise ValueError('Task/group overlap in follow-on')
        seen_keys.add(key); seen_groups.add(group)
    return selected


def previous_unstarted():
    # A result-free reserved request must never be silently redispatched.
    with (PREVIOUS / 'RUN.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = json.loads((PREVIOUS / 'config.json').read_text())
        if gen.base.sha(PREVIOUS / 'config.json') != (PREVIOUS / 'CONFIG.sha256').read_text().strip():
            raise ValueError('Predecessor config changed')
        for p, h in cfg['frozen_files'].items():
            if gen.base.sha(p) != h:
                raise ValueError('Predecessor frozen source changed')
        completed, sources = set(), []
        ledger = gen.base.read_rows(PREVIOUS / 'budget/dispatch.jsonl')
        settled = set()
        for role in ('gemini', 'k3'):
            calls = list((PREVIOUS / role / 'calls').glob('*.json'))
            attempts = list((PREVIOUS / role / 'attempts').glob('*.json'))
            if len(attempts) != len(calls):
                raise ValueError('Predecessor has unfinished requests')
            for p in attempts:
                a = json.loads(p.read_text())
                if a['state'] != 'settled' or gen.base.sha(a['receipt']) != a['receipt_sha256']:
                    raise ValueError('Predecessor request unresolved')
                settled.add(p.stem)
            for p in (PREVIOUS / role / 'results').glob('*.json'):
                completed.add(json.loads(p.read_text())['item']['item_id'])
                sources.append(p)
        if settled != {r['request_sha256'] for r in ledger}:
            raise ValueError('Predecessor dispatch/receipt mismatch')
        if len(completed) != 49 or len(settled) != 112:
            raise ValueError('Predecessor inventory changed; reconcile first')
    items = gen.base.read_rows(PREVIOUS / 'items.jsonl')
    baseline = gen.base.read_rows(BASELINE / 'MANIFEST.jsonl')
    if len(baseline) != 3979:
        raise ValueError('Accepted baseline changed')
    selected = choose_unstarted(items, completed, baseline)
    if len(selected) != 111:
        raise ValueError('Expected exactly 111 never-dispatched tasks')
    return selected, cfg, sources


def prepare(out):
    if out.exists():
        raise ValueError('Fresh output directory required')
    items, previous_cfg, sources = previous_unstarted()
    old_private = {r['item_id']: r for r in json.loads((PREVIOUS / 'SELECTION_PRIVATE.json').read_text())}
    private = [dict(old_private[i['item_id']], previous_assigned_role='gemini', assigned_role='k3',
                    reassignment='user authorized K3 continuation; no predecessor dispatch') for i in items]
    out.mkdir(parents=True)
    (out / 'items.jsonl').write_text(''.join(json.dumps(i, ensure_ascii=False) + '\n' for i in items))
    gen.write_json(out / 'SELECTION_PRIVATE.json', private)
    examples = []
    for src in map(Path, previous_cfg['examples']):
        dest = out / 'examples' / src.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, dest); examples.append(str(dest))
    spec = dict(previous_cfg['models']['k3'], tasks=111, call_cap=1110)
    cfg = dict(previous_cfg, version='sft-v2-topup-k3-followon-20261008', created_at=time.time(),
               deadline=time.time()+8*3600, tasks=111, workers=2, examples=examples,
               models={'k3': spec}, baseline_T_Q_G=dict(T=3979, Q=3979, G=3629),
               minimum_new_KEEP_for_over_4000=22,
               authorization='2026-10-08 user: 继续使用k3来补充; continue the never-dispatched remainder, Codex reviews as before',
               predecessor=str(PREVIOUS), baseline_manifest=str(BASELINE/'MANIFEST.jsonl'),
               budget_units='1110 client HTTP requests; no AGY/Claude/Astra calls; no retries or account switching')
    brief = '''# V2.1 后续补充：K3 接替未派发队列

用户在本轮明确要求“继续使用k3来补充”。基线为 T/Q/G 3979/3979/3629，超过4000还差22条KEEP。
仅将上一批分给Gemini且从未派发的111题转给K3；排除Gemini已失败的首题及K3已完成48题。
111题对应111个新原图组：旧WEAK 97、BAD 6、UNRESOLVED 8。限定未决题仅尝试，不假定可回答。
MME 44、WorldBench 26、MMMU 17、VisReason 15、V* 9。与基线/示范按任务与原图组零重叠。

## 冻结执行与边界

cwd=${ICML_DIR}；独立run，原run/STOP/标签/GT/3953与3979导出不改。
仅K3-256K low，沿用现有LiteLLM本地端点；2并发、12客户端请求/分钟、1110 HTTP请求总帽、8小时墙钟。
每题至多8次crop和终答，最多1次格式修复；有效错答/弃答/截断/技术错误不重刷，不回放未确认请求。
先用队首真实任务核对模型回执，成功后继续固定题单。没有额外连通题、没有其他模型或账号。
不新增题，不调用Gemini/AGY/Astra/Claude，不恢复旧K3审查/GPU/训练。
继承原Caption、两份真实完整示范、生产缩放与真实crop。原图足够可直答，不强行多步。
GT、旧轨迹和旧审核理由仅离线保存，不进入生成上下文。

## 监控与停止

依据results/calls/attempts/budget/worker.log，不能凭PID。每次派发核对源/图像哈希。
PAUSE、源变化、截止停止新派发；认证/额度/429、身份异常、连续3次技术失败停止K3。
保留在途回执，意外退出先查attempts与RUN.lock，只报告恢复位置，不自动重启。
启动并有真实回执后恢复同一icml heartbeat（15分钟）；正常推进静默，仅完成/失败/停滞30分钟/需决策通知。

## Codex审查与交付

所有终答正确且CPU合格的新路径由Codex沿用已授权的逐图规则审核；可沿用独立native reviewer分队。
先看原图盲答，再看整条路径/实际crop；核查前缀可见证据、关键事实、对象绑定及每次crop用途。
只收完整KEEP：允许有效负证据和诚实未命中，不收纯重复放大、未修复关键错误或问题未决。
WEAK/BAD/UNRESOLVED独立留档，不改旧标签，不拼接路径，不冒称人工金标准或多模型族共识。
新KEEP通过CPU/导出/监督mask后，按任务与原图组去重形成独立增量版；报告T/Q/G及距4001缺口。
沿用公开仓库脱敏边界更新结果。处理完这111题即收口；不足也不自动扩题或重刷。
'''
    (out / 'WORK_PACKAGE.zh.md').write_text(brief)
    extra = [Path(__file__), PREVIOUS/'config.json', PREVIOUS/'CLOSEOUT.json', PREVIOUS/'items.jsonl',
             PREVIOUS/'SELECTION_PRIVATE.json', BASELINE/'MANIFEST.jsonl', BASELINE/'caption_sft_v2_1_keep.json',
             out/'items.jsonl', out/'SELECTION_PRIVATE.json', out/'WORK_PACKAGE.zh.md', *map(Path, examples), *sources]
    cfg['frozen_files'] = dict(previous_cfg['frozen_files'])
    cfg['frozen_files'].update({str(p):gen.base.sha(p) for p in extra})
    gen.write_json(out/'config.json', cfg)
    (out/'CONFIG.sha256').write_text(gen.base.sha(out/'config.json')+'\n')
    gen.write_json(out/'SELECTION_STATS.json', dict(tasks=len(items), unique_groups=len(items),
        prior_verdicts=dict(Counter(r['prior_verdict'] for r in private)),
        prior_crops=dict(Counter(r['prior_crops'] for r in private)),
        benchmarks=dict(Counter(i['bench'] for i in items)),
        materials=dict(Counter(i['material'] for i in items)), model_tasks={'k3':111},
        baseline_task_overlap=0, baseline_group_overlap=0, demo_group_overlap=0,
        excludes_all_predecessor_dispatched_tasks=True))
    verify(out)
    return status(out)


def verify(out):
    cfg, examples, items = gen.base.verify(out)
    if list(cfg['models']) != ['k3'] or any(i['assigned_role'] != 'k3' for i in items):
        raise ValueError('K3-only dispatch required')
    baseline = gen.base.read_rows(Path(cfg['baseline_manifest']))
    eligible, _, _ = previous_unstarted()
    if items != eligible or len(items) != 111 or len(baseline) != 3979:
        raise ValueError('Follow-on scope differs from never-dispatched inventory')
    return cfg, examples, items


def status(out):
    data = ORIGINAL_STATUS(out)
    cfg = json.loads((out/'config.json').read_text())
    data['baseline_accepted_SFT'] = cfg['baseline_T_Q_G']['T']
    data['minimum_new_KEEP_for_over_4000'] = 22
    gen.write_json(out/'STATUS.json',data)
    return data


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','run','status','handoff'])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if a.action=='prepare':print(json.dumps(prepare(out),ensure_ascii=False))
    elif a.action=='verify':cfg,_,items=verify(out);print(json.dumps(dict(valid=True,tasks=len(items))))
    elif a.action=='status':print(json.dumps(status(out),ensure_ascii=False))
    elif a.action=='handoff':print(json.dumps(gen.base.handoff(out),ensure_ascii=False))
    else:
        gen.verify=verify;gen.status=status;gen.run(out)
