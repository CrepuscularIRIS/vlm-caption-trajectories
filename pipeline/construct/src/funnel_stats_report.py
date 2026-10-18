"""Funnel statistics in the mentor's units (2026-10-06; reports/mentor_meeting_20260928/09).

A trajectory = one model's complete interaction path for one task (one results/*.json); steps, calls and crops never
count as trajectories. Final training scale is reported as T (accepted complete trajectories, one per task),
Q (deduplicated image+question tasks) and G (original image groups), computed only from an accepted SFT list.
Before that list exists this report gives the candidate-stage numbers and says so. Reads select_v2 (features of every
result file, CPU gates, per-item consensus) and the audit sidecars; no model calls.
"""
import argparse
import json
import logging
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from funnel_select import gate

logger = logging.getLogger(__name__)

NAME = {'gpt-6-luna': 'Luna (gpt-6-luna)', 'glm-5.3-flash': 'GLM 5.3 Flash', 'gpt-5.6-sol': 'Sol 5.6',
        'grok-4.7': 'Grok 4.7（Cursor+xAI 两路，同一模型）', 'gpt-6.1-sol': 'Sol 6.1（已停，部分）',
        'k3-256k': 'K3（部分）', 'kimi-for-coding': 'K2.8（部分）'}
FAMILY = {'gpt-6-luna': 'openai', 'gpt-5.6-sol': 'openai', 'glm-5.3-flash': 'zhipu', 'grok-4.7': 'xai'}
SAME = {'grok-4.7-cpa': 'grok-4.7', 'grok-4.7-cur': 'grok-4.7'}


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def per_model(trajs: list[dict]) -> dict:
    out = {}
    by = defaultdict(list)
    for t in trajs:
        by[SAME.get(t['model'], t['model'])].append(t)
    for m, ts in by.items():
        term = Counter(t['term'] for t in ts)
        correct = [t for t in ts if t['term'] == 'answer' and t['score'] == 1]
        wrong = sum(t['term'] == 'answer' and t['score'] == 0 for t in ts)
        gates = Counter(g for t in correct for g in gate(t))
        qual = [t for t in correct if not gate(t)]
        out[m] = dict(
            name=NAME.get(m, m), records=len(ts), tasks=len({t['item_id'] for t in ts}),
            terminations=dict(term), correct=len(correct), wrong=wrong,
            answered_unscored=sum(t['term'] == 'answer' and t['score'] is None for t in ts),
            accuracy_of_records=round(len(correct) / len(ts), 3),
            gate_fail_among_correct=dict(gates), qualified=len(qual),
            qualified_groups=len({t['group_id'] for t in qual}),
            direct_share=round(sum(t['crops'] == 0 for t in ts) / len(ts), 3),
            steps_median=st.median(t['n_steps'] for t in ts), steps_mean=round(st.mean(t['n_steps'] for t in ts), 2),
            crops_mean=round(st.mean(t['crops'] for t in ts), 2),
            completion_tokens_median=st.median(t['completion_tokens'] or 0 for t in ts),
            total_tokens_median=st.median(t['tokens'] or 0 for t in ts),
            qualified_steps_median=st.median(t['n_steps'] for t in qual) if qual else 0,
            qualified_completion_tokens_median=st.median(t['completion_tokens'] or 0 for t in qual) if qual else 0,
            by_bench={b: dict(records=sum(t['bench'] == b for t in ts),
                              correct=sum(t['bench'] == b and t['term'] == 'answer' and t['score'] == 1 for t in ts),
                              qualified=sum(t['bench'] == b for t in qual))
                      for b in sorted({t['bench'] for t in ts})})
    return out


def task_level(trajs: list[dict], items: list[dict]) -> dict:
    qual_fams = defaultdict(set)
    for t in trajs:
        m = SAME.get(t['model'], t['model'])
        if m in FAMILY and t['term'] == 'answer' and t['score'] == 1 and not gate(t):
            qual_fams[t['item_id']].add(FAMILY[m])
    k = Counter(len(qual_fams.get(i['item_id'], ())) for i in items)
    groups = defaultdict(set)
    mats = defaultdict(Counter)
    for i in items:
        n = len(qual_fams.get(i['item_id'], ()))
        groups[min(n, 2)].add(i['group_id'])
        mats[i['material']]['>=2' if n >= 2 else str(n)] += 1
    vision = [i for i in items if i['item_id'].startswith('mmmu_pro_vision')]
    return dict(tasks=len(items), groups=len({i['group_id'] for i in items}),
                tasks_by_families_with_qualified=dict(sorted(k.items())),
                groups_with_2plus_families=len(groups[2]),
                tasks_by_material=dict(mats), by_bench=dict(Counter(i['bench'] for i in items)),
                mmmu_vision_tasks_source_unverified=len(vision),
                mmmu_vision_with_2plus=sum(len(qual_fams.get(i['item_id'], ())) >= 2 for i in vision),
                gold_suspect=sum(bool(i.get('gold_suspect')) for i in items))


def audits(runs: Path) -> dict:
    out = {}
    luna = json.loads((runs / 'audit_luna_v1_20261003' / 'POOLS.json').read_text())
    out['luna_v1'] = dict(rows=len(luna['rows']), pools=dict(Counter(luna['rows'].values())),
                          clean_groups=luna.get('clean_groups'))
    cons = runs / 'audit_consensus_v1_20261006'
    sides = [json.loads(p.read_text()) for p in sorted((cons / 'audit').glob('cons_*.json'))]
    manifest = sum(1 for _ in (cons / 'manifest.jsonl').open())
    dec = Counter(s['decision'] for s in sides)
    by_model = defaultdict(Counter)
    for s in sides:
        by_model[SAME.get(s['model'], s['model'])][s['decision']] += 1
    out['consensus_t1'] = dict(to_audit=manifest, audited=len(sides), decisions=dict(dec),
                               by_model={m: dict(c) for m, c in by_model.items()},
                               reused_luna_clean=sum(1 for _ in (cons / 'reused.jsonl').open()),
                               selection=json.loads((cons / 'SELECTION.json').read_text())['chosen_by_material'])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--funnel', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    sel = a.funnel / 'select_v2'
    # the cache is append-on-change: keep the last row per result file, and only files that still exist
    latest = {}
    for line in (sel / 'traj_cache.jsonl').open():
        row = json.loads(line)
        latest[row['path']] = row['f']
    trajs = [f for path, f in latest.items() if Path(path).exists()]
    items = [json.loads(l) for l in (sel / 'items.jsonl').open()]
    pool_rows = sum(1 for _ in (a.funnel.parent.parent / 'benchmarks' / 'funnel_v1_20261002' / 'pool.jsonl').open())
    stats = dict(pool_rows=pool_rows, trajectories_total=len(trajs), per_model=per_model(trajs),
                 task_level=task_level(trajs, items), audits=audits(a.funnel.parent),
                 final_TQG='not yet: no accepted SFT list; numbers below are candidate stage')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'STATS.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    logger.info(json.dumps(stats, ensure_ascii=False)[:200])


if __name__ == '__main__':
    main()
