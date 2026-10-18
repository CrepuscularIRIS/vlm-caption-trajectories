"""Multi-step-first swap proposal (~4,000 tasks, one complete path each) from the unified KEEP inventory (2026-10-08).

CPU only; no model calls; does not touch the exported 4k or any label. Input: sft_comprehensive_audit_20261008/
KEEP_PRIORITY_INDEX.jsonl (stored KEEP, no recorded audit flag, export regex clean) joined with the per-path review rows for
observed_behavior / imitate. Per task: a useful tool path beats a direct path; a tool path whose crops the reviewer called
unnecessary or partly useless does not. Across tasks: every task with a useful tool path, then direct controls up to --target.
"""
import argparse
import collections
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path('${ICML_DIR}')
AUD = ROOT / 'reports/sft_comprehensive_audit_20261008'
RUNS = ROOT / 'obs/runs'
SCORE_FILES = ['legacy_pass_rescore_20261007/SCORES.jsonl', 'scorer_fix_review_20261007/SCORES.jsonl', 'pool_scoring_20261007/LAYER1_SCORES.jsonl',
               'pool_scoring_20261007/rescored_visreason/SCORES.jsonl', 'probe_ladder_20261007/ASTRA_SCORES.jsonl',
               'gen_regen_gemini_20261007/review/REGEN_SCORES.jsonl', 'gen_regen_gemini_20261007/review_nocand/SCORES.jsonl',
               'sft_multistep_regen_20261007/review_opus/SCORES.jsonl']
DEMO_ITEMS = {'mme_realworld_10768', 'mme_realworld_lite_10768', 'mme_realworld_20370', 'mme_realworld_09808', 'worldbench_1844'}
DIRECT_QUOTA = {'mmmu_pro': .30, 'worldbench': .20, 'visreason': .10, 'vstar': .03, 'mme_realworld': .37}


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def detail() -> dict[str, dict]:
    out = {}
    for f in SCORE_FILES:
        for line in (RUNS / f).open():
            d = json.loads(line)
            if d.get('result_path'):
                out[d['result_path']] = dict(observed=d.get('observed_behavior'), imitate=d.get('imitate'))
    return out


def tool_class(r: dict) -> str:
    """useful_multi / useful_single / doubtful_tool / direct / direct_needed_zoom"""
    if r['crops'] == 0:
        return 'direct_needed_zoom' if r['observed'] == 'direct_but_needed_zoom' else 'direct'
    if r['observed'] == 'unnecessary_crops' or r.get('crops_informative') in ('some_useless', 'mostly_useless') or r['redundancy'] == 'high':
        return 'doubtful_tool'
    return 'useful_multi' if r['crops'] >= 2 else 'useful_single'


RANK = {'useful_multi': 0, 'useful_single': 0, 'direct': 1, 'doubtful_tool': 2, 'direct_needed_zoom': 3}


def key(r: dict) -> tuple:
    c = r['cls']
    return (RANK[c], -(r['imitate'] or 3), r['redundancy'] != 'low', -(r['evidence'] or 3) - (r['binding'] or 3) - (r['action_fit'] or 3),
            -min(r['crops'], 3) if c.startswith('useful') else 0, h(r['result_path']))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--target', type=int, default=4000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    det = detail()
    by_task = collections.defaultdict(list)
    for line in (AUD / 'KEEP_PRIORITY_INDEX.jsonl').open():
        r = json.loads(line)
        if r['item_id'] in DEMO_ITEMS:
            continue
        r.update(det.get(r['result_path'], dict(observed=None, imitate=None)))
        r['cls'] = tool_class(r)
        by_task[r['canonical_sample_key']].append(r)
    best = {}
    for k, rows in by_task.items():
        rows.sort(key=key)
        b = dict(rows[0])
        b['n_alternatives'] = len(rows) - 1
        b['alternative_classes'] = sorted({x['cls'] for x in rows[1:]})
        best[k] = b
    tool = [b for b in best.values() if b['cls'].startswith('useful')]
    rest = [b for b in best.values() if not b['cls'].startswith('useful')]
    sel, seen_g = list(tool), {g for b in tool for g in b['canonical_groups']}
    need = max(0, a.target - len(sel))
    # direct controls: clean direct first, new image groups first, spread over benchmarks
    rest.sort(key=lambda b: (RANK[b['cls']], any(g in seen_g for g in b['canonical_groups']), -(b['imitate'] or 3), h(b['result_path'])))
    by_bench = collections.defaultdict(list)
    for b in rest:
        by_bench[b['bench']].append(b)
    picked = []
    for bench, share in DIRECT_QUOTA.items():
        picked += by_bench[bench][:round(need * share)]
        by_bench[bench] = by_bench[bench][round(need * share):]
    left = sorted((b for v in by_bench.values() for b in v), key=lambda b: (RANK[b['cls']], h(b['result_path'])))
    picked += left[:need - len(picked)]
    sel += picked[:need]
    held = [b for b in rest if b not in picked[:need]]
    a.out.mkdir(parents=True, exist_ok=True)
    keep = ('item_id', 'canonical_sample_key', 'canonical_groups', 'bench', 'material', 'model', 'stage', 'result_path', 'result_sha256', 'crops',
            'steps', 'cls', 'observed', 'imitate', 'redundancy', 'review_source', 'selection_status', 'n_alternatives', 'alternative_classes')
    for name, rows in (('SELECTION_PROPOSAL.jsonl', sel), ('NOT_SELECTED_KEEP_TASKS.jsonl', held)):
        (a.out / name).write_text(''.join(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + '\n' for r in rows))
    C = collections.Counter
    stats = dict(tasks_with_priority_keep=len(best), selected=len(sel), groups=len({g for b in sel for g in b['canonical_groups']}),
                 by_class=dict(C(b['cls'] for b in sel)), by_model=dict(C(b['model'] for b in sel)), by_bench=dict(C(b['bench'] for b in sel)),
                 by_model_class={f"{m}|{c}": n for (m, c), n in sorted(C((b['model'], b['cls']) for b in sel).items())},
                 by_review_source=dict(C(b['review_source'] for b in sel)),
                 vs_old_4k=dict(C('same_path_as_old_4k' if b['selection_status'] == 'selected_4k' else 'new_or_swapped' for b in sel)),
                 not_selected=dict(C(b['cls'] for b in held)))
    (a.out / 'STATS.json').write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    logger.info(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
