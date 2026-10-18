"""4,000-task selection list (2026-10-07; user: keep only clean paths, WEAK / rejected tasks go to regeneration).

CPU only, no model calls, v0.3 and audits untouched. Clean candidate per task, in priority order:
  1. v0.3 path (earlier model audit pass), after the 174-task closeout: swaps applied, hold dropped, closeout WEAK moved out
  2. layer-1 Sonnet score KEEP (dispute adjudication overrides), best per task by imitate
  3. reserve: earlier model audit AUTO_PASS path not in v0.3
Per-bench soft quota (MME 2400, MMMU 800, WB 600, VR 160, V* 40). When a bench has more clean tasks than its quota, fill
greedily by behavior deficit against BEHAVIOR_TARGET, new image groups first, then imitate, then hash. Shortfall is reported,
never filled with WEAK. Outputs in --out: SELECTION_4K.jsonl, RESERVE_CLEAN.jsonl, REGEN_CANDIDATES.jsonl, HELD.jsonl, STATS.json.
"""
import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path

from export_protocol_check import HOLD, check_record

ROOT = Path('${ICML_DIR}')
SC = ROOT / 'obs/runs/pool_scoring_20261007'
QUOTA = {'mme_realworld': 2400, 'mmmu_pro': 800, 'worldbench': 600, 'visreason': 160, 'vstar': 40}
BEHAVIOR_TARGET = {'direct': .25, 'single_view_binding': .35, 'multi_view_evidence': .25, 'search_or_switch': .15}
BAD_ANSWERABILITY = {'object_ambiguous', 'question_intent_unclear', 'gt_doubtful', 'not_visible'}


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def rp(p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else (ROOT / 'obs/src' / q).resolve())


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--extra-reserve', type=Path, nargs='*', default=[], help='KEEP files from the generation ladder (Gemini / Astra / rescored)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    paths = {r['result_path']: r for r in jl(SC / 'PATH_SCORES.jsonl')}
    tasks = {t['item_id']: t for t in jl(SC / 'TASK_SCORES.jsonl')}
    by_task_paths = collections.defaultdict(list)
    for r in paths.values():
        by_task_paths[r['item_id']].append(r)
    closeout = {json.loads(Path(f).read_text())['item_id']: json.loads(Path(f).read_text())
                for f in glob.glob(str(ROOT / 'obs/runs/closeout_review_20261007/records/*.json'))}
    dispute = {d['key']: d for d in json.loads((SC / 'LAYER1_DISPUTE_RESULTS.json').read_text())}
    scores = collections.defaultdict(list)
    for r in jl(SC / 'LAYER1_SCORES.jsonl'):
        if r['key'] in dispute:
            r = dict(r, verdict=dispute[r['key']]['final'], adjudicated=True)
        scores[r['item_id']].append(r)

    clean, regen, held = {}, [], []
    diag = set((ROOT / 'obs/runs/audit_consensus_v1_20261006/DIAGNOSTIC_GT_OPTION_ITEMS.txt').read_text().split())

    def add(item_id: str, path: str, source: str, verdict: str, imitate, note: str) -> None:
        if item_id in diag:  # GT / option problem found in the 192 spot check
            held.append(dict(item_id=item_id, reason='diagnostic_gt_option_problem', detail=source))
            return
        p = paths.get(path, {})
        t = tasks[item_id]
        clean[item_id] = dict(item_id=item_id, bench=t['bench'], material=t['material'], group_id=t['group_id'], model=p.get('model'),
                              result_path=path, behavior=p.get('behavior'), source=source, verdict=verdict, imitate=imitate, note=note)

    for r in jl(ROOT / 'obs/benchmarks/sft_v0_3_20261006/MANIFEST.jsonl'):
        co = closeout.get(r['item_id'])
        if co:
            if co['action'] == 'unresolved_hold':
                held.append(dict(item_id=r['item_id'], reason='closeout_unresolved', detail=co.get('adjudication_reason', '')[:300]))
                continue
            if co.get('chosen_verdict') == 'WEAK':
                regen.append(dict(item_id=r['item_id'], bench=tasks[r['item_id']]['bench'], reason='closeout_weak', best_verdict='WEAK'))
                continue
            if co['action'] == 'swap':
                add(r['item_id'], rp(co['chosen_result_path']), 'v03_closeout_swap', co['chosen_verdict'], None, 'closeout swap')
                continue
        add(r['item_id'], rp(r['result_path']), 'v03_audit_pass', 'AUTO_PASS', None, r.get('decision', ''))

    for item_id, rs in scores.items():
        if item_id in clean or any(x['item_id'] == item_id for x in regen):
            continue
        keeps = [x for x in rs if x['verdict'] == 'KEEP']
        if keeps:
            b = max(keeps, key=lambda x: (x['imitate'], -h(x['key']).__hash__() % 1))
            add(item_id, b['result_path'], 'layer1_keep', 'KEEP', b['imitate'], b['observed_behavior'])
            continue
        ans = collections.Counter(x['answerability'] for x in rs)
        if ans and ans.most_common(1)[0][0] in BAD_ANSWERABILITY:
            held.append(dict(item_id=item_id, reason='answerability_' + ans.most_common(1)[0][0], detail=rs[0]['issue'][:300]))
            continue
        best = 'WEAK' if any(x['verdict'] == 'WEAK' for x in rs) else 'BAD_OR_UNRESOLVED'
        regen.append(dict(item_id=item_id, bench=tasks[item_id]['bench'], reason='layer1_no_keep', best_verdict=best,
                          n_scored=len(rs), verdicts=dict(collections.Counter(x['verdict'] for x in rs))))

    for item_id, t in tasks.items():
        if item_id in clean:
            continue
        if t['task_class'] == 'reserve_has_pass':
            ok = [p for p in by_task_paths[item_id] if p['audit'] == 'pass' and p['cpu_candidate']]
            if ok:
                add(item_id, ok[0]['result_path'], 'reserve_audit_pass', 'AUTO_PASS', None, 'earlier audit pass, not in v0.3')
        elif t['task_class'] in ('all_candidates_rejected', 'dispute_rejected_and_disputed', 'dispute_only_disputed') and item_id not in scores:
            regen.append(dict(item_id=item_id, bench=t['bench'], reason='all_audited_paths_rejected_or_disputed', best_verdict='REJECTED'))

    obs2beh = {'direct_sufficient': 'direct', 'direct_but_needed_zoom': 'direct', 'local_recognition': 'single_view_binding',
               'multi_place_evidence': 'multi_view_evidence', 'search_then_found': 'search_or_switch', 'search_miss_reported': 'search_or_switch',
               'unnecessary_crops': 'multi_view_evidence'}
    for fx in a.extra_reserve:
        for r in jl(fx):
            if r['item_id'] in clean or r['item_id'] in diag:
                continue
            t = tasks[r['item_id']]
            beh = paths.get(r['result_path'], {}).get('behavior') or obs2beh.get(r.get('observed_behavior'), 'single_view_binding')
            clean[r['item_id']] = dict(item_id=r['item_id'], bench=t['bench'], material=t['material'], group_id=t['group_id'], model=r['model'],
                                       result_path=r['result_path'], behavior=beh, source=r['source'], verdict='KEEP', imitate=r.get('imitate'),
                                       note='generation ladder')
            regen = [x for x in regen if x['item_id'] != r['item_id']]

    # export protocol gate (caption coordinates / layout) before quota fill, so held paths never take a slot
    for item_id in list(clean):
        rec = json.loads(Path(clean[item_id]['result_path']).read_text())['rollout']['export']
        bad = sorted(set(check_record(rec)) & HOLD)
        if bad:
            held.append(dict(item_id=item_id, reason='export_protocol_' + '+'.join(bad), detail=clean[item_id]['result_path']))
            del clean[item_id]

    # quota fill
    sel, reserve = [], []
    for b, q in QUOTA.items():
        pool = [c for c in clean.values() if c['bench'] == b]
        prio = collections.defaultdict(lambda: 3, {'v03_audit_pass': 0, 'v03_closeout_swap': 0, 'layer1_keep': 1, 'reserve_audit_pass': 2})
        if len(pool) <= q:
            sel += pool
            continue
        counts, groups, chosen = collections.Counter(), set(), []
        rest = sorted(pool, key=lambda c: (prio[c['source']], -(c['imitate'] or 0), h(c['item_id'])))
        while len(chosen) < q and rest:
            n = len(chosen) + 1
            def key(c):
                deficit = BEHAVIOR_TARGET.get(c['behavior'], 0) * n - counts[c['behavior']]
                return (c['group_id'] in groups, -round(deficit, 3), prio[c['source']], -(c['imitate'] or 0), h(c['item_id']))
            c = min(rest, key=key)
            rest.remove(c)
            chosen.append(c)
            counts[c['behavior']] += 1
            groups.add(c['group_id'])
        sel += chosen
        reserve += rest

    def w(name, rows):
        with (a.out / name).open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    w('SELECTION_4K.jsonl', sel)
    w('RESERVE_CLEAN.jsonl', reserve)
    w('REGEN_CANDIDATES.jsonl', regen)
    w('HELD.jsonl', held)
    C = collections.Counter
    stats = dict(T=len(sel), Q=len({s['item_id'] for s in sel}), G=len({s['group_id'] for s in sel}),
                 by_bench={b: dict(selected=sum(s['bench'] == b for s in sel), quota=q, clean_available=sum(c['bench'] == b for c in clean.values()))
                           for b, q in QUOTA.items()},
                 by_source=dict(C(s['source'] for s in sel)), by_behavior=dict(C(s['behavior'] for s in sel)), by_model=dict(C(s['model'] for s in sel)),
                 by_material=dict(C(s['material'] for s in sel)), reserve_clean=len(reserve),
                 regen=dict(total=len(regen), by_reason=dict(C(r['reason'] for r in regen)), by_bench=dict(C(r['bench'] for r in regen)),
                            by_best=dict(C(r['best_verdict'] for r in regen))),
                 held=dict(total=len(held), by_reason=dict(C(x['reason'] for x in held))))
    (a.out / 'STATS.json').write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
