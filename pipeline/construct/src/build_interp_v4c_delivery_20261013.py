"""Assemble the two v4c libraries and export them to the private data repository. Read-only over runs.

Both libraries use caption v6, the first library's 2 MP view delivery and a crop budget of 8, the same
teacher-side condition and the same generator order (GPT-5.6 Sol, then GPT-6.1 Sol, then GPT-6 Astra):
  interp   forced misses on the generator's own crop requests (variant ctrl_2mp_b8), those turns at loss 0;
  control  no forced miss (variant ctrl_2mp_b8_nomiss).
Per task the first accepted episode in generator order is taken; every saved episode of a model counts as
its one attempt. Accepted = ``interp_v4c_select_20261013.accepted`` (frozen scorer plus four added rules);
it is an objective check, not a semantic or pixel acceptance. GT is used here for offline selection only.
All splits go into the training files (user decision 2026-10-13, as in the first SFT); the split stays in
the database rows. A matched subset holds the tasks present in both libraries.
"""
import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import caption_v6 as proto
import interp_v4c_select_20261013 as select

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'obs/runs'
OUT = ROOT/'deliverables/interp_v4c_20261013'
REPO = Path('${DATA_DIR}/vlm-sft-v2-2')
PRIVATE = RUNS/'interp_v4c_sol56_20261013/SELECTION_PRIVATE.json'
PILOT_PRIVATE = RUNS/'interp_v4c_sol56_pilot1_20261013/SELECTION_PRIVATE.json'
LIBRARIES = dict(
    interp=dict(variant='ctrl_2mp_b8', pools=[
        'interp_v4c_sol56_pilot1_20261013', 'interp_v4c_sol56_20261013', 'interp_v4c_sol56_laneA_20261013',
        'interp_v4c_sol56_laneB_20261013', 'interp_v4c_sol56_rest_20261013',
        'interp_v4c_sol61_pilot1_20261013', 'interp_v4c_sol61_fill_20261013',
        'interp_v4c_pilot1_20261013', 'interp_v4c_astra_20261013', 'interp_v4c_astra_fill_20261013']),
    control=dict(variant='ctrl_2mp_b8_nomiss', pools=[
        'interp_v4c_nomiss_sol56_20261013', 'interp_v4c_nomiss_sol61_fill_20261013', 'interp_v4c_nomiss_astra_fill_20261013']))
MODEL_ORDER = ['gpt-5.6-sol', 'gpt-6.1-sol', 'gpt-6-astra']
LENGTH_BINS = ((1, 1, '1'), (2, 2, '2'), (3, 3, '3'), (4, 5, '4-5'), (6, 9, '6-9'))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(name: str, allow_running: bool) -> Dict[str, List[dict]]:
    """Saved episodes per task in generator order, each with the rule that accepts it (or None)."""
    spec, by_task = LIBRARIES[name], collections.defaultdict(list)
    for pool in spec['pools']:
        parent = RUNS/pool
        for state in parent.glob('continuations/*/sol/RUN_STATE.json'):
            if json.loads(state.read_text())['state'] == 'running' and not allow_running:
                raise ValueError('wave still running: ' + str(state.parent))
        for e in select.episodes(parent):
            result = e['result']
            if (result.get('construction') or {}).get('variant') not in (spec['variant'], None):
                raise ValueError(f"{pool}/{e['audit_id']}: unexpected variant")
            by_task[e['audit_id']].append(dict(e, campaign=pool))
    for attempts in by_task.values():
        attempts.sort(key=lambda e: MODEL_ORDER.index(e['result']['model']))
    return by_task


def row(private: dict, e: dict) -> dict:
    result = e['result']
    ro, c = result['rollout'], result['construction']
    export, steps = ro['export'], ro['steps']
    if export['conversations'][0]['value'] != proto.SYSTEM or export['protocol'] != proto.PROTOCOL_VERSION:
        raise ValueError('Not a caption v6 export: ' + result['audit_id'])
    if c['view_max_side'] is not None or export['budget'] != 8:
        raise ValueError('Not first-library delivery and budget: ' + result['audit_id'])
    forced = [s for s in steps if s['origin'] == 'editor_forced']
    if export['assistant_supervision'] != [s['origin'] != 'editor_forced' for s in steps]:
        raise ValueError('Supervision flags do not match step origins: ' + result['audit_id'])
    return dict(
        audit_id=result['audit_id'], item_id=private['manifest']['item_id'], split=private['split'],
        bench=private['manifest'].get('bench'), material=private['manifest'].get('material'),
        canonical_groups=private['manifest']['canonical_groups'], generator=result['model'], campaign=e['campaign'],
        accepted_by=e['rule'], source_result=str(e['path'].relative_to(ROOT)), source_result_sha256=sha(e['path']),
        conversations=export['conversations'], images=export['images'], assistant_supervision=export['assistant_supervision'],
        protocol=export['protocol'], budget=export['budget'], arm=c['arm'], variant=c['variant'],
        steps=len(steps), crops=ro['crops'], skills=c['skills'], verdicts=c['verdicts'],
        induced_misses=len(forced), induced_kinds=c['kinds'], induced_steps=c['intervention_steps'], reactions=c['reactions'],
        objective_candidate=True, semantic_review='NOT_DONE', target_4B_loader_verified=False)


def assemble(name: str, private: Dict[str, dict], allow_running: bool) -> Tuple[List[dict], List[dict]]:
    by_task, rows, ledger = episodes(name, allow_running), [], []
    for aid in sorted(private):
        attempts = by_task.get(aid, [])
        chosen = next((e for e in attempts if e['rule'] and e['result']['rollout'].get('export')), None)
        ledger.append(dict(audit_id=aid, split=private[aid]['split'], in_library=chosen is not None,
                           generator=chosen['result']['model'] if chosen else None,
                           attempts=[dict(campaign=e['campaign'], model=e['result']['model'],
                                          termination=e['result']['rollout']['termination'], accepted_by=e['rule']) for e in attempts]))
        if chosen:
            rows.append(row(private[aid], chosen))
    return rows, ledger


def summary(rows: List[dict], ledger: List[dict]) -> dict:
    count = collections.Counter

    def table(part: List[dict]) -> dict:
        steps = sorted(r['steps'] for r in part)
        return dict(tasks=len(part), mean_steps=round(sum(steps) / max(1, len(steps)), 3),
                    median_steps=steps[len(steps) // 2] if steps else None,
                    steps=dict(sorted(count(steps).items())),
                    length_bins={n: sum(lo <= s <= hi for s in steps) for lo, hi, n in LENGTH_BINS},
                    supervised_turns=sum(sum(r['assistant_supervision']) for r in part), assistant_turns=sum(steps),
                    with_induced_miss=sum(r['induced_misses'] > 0 for r in part), induced_misses=sum(r['induced_misses'] for r in part))

    def group(key) -> dict:
        return {str(k): table([r for r in rows if key(r) == k]) for k in sorted({key(r) for r in rows}, key=str)}

    reactions = [x for r in rows for x in r['reactions']]
    return dict(
        tasks_total=len(ledger), in_library=len(rows), missing=len(ledger) - len(rows),
        never_attempted=sum(not l['attempts'] for l in ledger),
        missing_by_last_outcome=dict(count(l['attempts'][-1]['termination'] for l in ledger if not l['in_library'] and l['attempts'])),
        accepted_by=dict(count(r['accepted_by'] for r in rows)), split=dict(count(r['split'] for r in rows)),
        all=table(rows), by_generator=group(lambda r: r['generator']), by_material=group(lambda r: r['material']),
        by_arm=group(lambda r: r['arm']), by_bench=group(lambda r: r['bench']),
        skill_turns=dict(count(s for r in rows for s in r['skills']).most_common()),
        induced_kinds=dict(count(k for r in rows for k in r['induced_kinds'])),
        reaction_verdict_by_kind={k: dict(count(str(x['next_verdict']) for x in reactions if x['kind'] == k)) for k in sorted({x['kind'] for x in reactions})},
        reaction_skill_by_kind={k: dict(count(str(x['next_skill']) for x in reactions if x['kind'] == k)) for k in sorted({x['kind'] for x in reactions})},
        caveats=['objective candidates only; no semantic or pixel acceptance',
                 'reaction tables are read from caption text and skill words, not checked against pixels',
                 'target Qwen3-VL-4B loader, masks, EOS, truncation and training not verified',
                 'caption v6 protocol: do not mix with v5.1 libraries in one training set',
                 'all splits are used for training in these files; the split field stays in the database rows'])


def repo_image(path: str, name: str, audit_id: str, repo: Optional[Path]) -> str:
    """Repository-relative path of a view; original images already in the repository are reused."""
    source = Path(path)
    try:
        shared = Path('images')/source.relative_to(RUNS)
    except ValueError:
        shared = None
    if shared is not None and '/interp_v4c_' not in str(source):
        if repo is None or (repo/shared).exists():
            return str(shared)
    target = Path('images/interp_v4c_20261013')/name/audit_id/source.name
    if repo is not None:
        (repo/target).parent.mkdir(parents=True, exist_ok=True)
        if not (repo/target).exists():
            shutil.copyfile(source, repo/target)
    return str(target)


def sharegpt(rows: List[dict], name: str, repo: Optional[Path]) -> List[dict]:
    return [dict(id=r['audit_id'], conversations=r['conversations'],
                 images=[repo_image(p, name, r['audit_id'], repo) for p in r['images']],
                 assistant_supervision=r['assistant_supervision']) for r in rows]


def write_json(path: Path, value: object, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, **(dict(separators=(',', ':')) if compact else dict(indent=1))))


def build(out: Path, repo: Optional[Path], allow_running: bool, names: List[str]) -> dict:
    private = {p['audit_id']: p for f in (PRIVATE, PILOT_PRIVATE) for p in json.loads(f.read_text())}
    if len(private) != 4040:
        raise ValueError('Task table is not the 4040 tasks')
    built = {name: assemble(name, private, allow_running) for name in names}
    paired = len(built) > 1  # the matched subset only exists when both libraries are assembled
    matched = sorted(set.intersection(*({r['audit_id'] for r in rows} for rows, _ in built.values()))) if paired else []
    report = dict(libraries=names, matched_tasks=len(matched) if paired else None)
    if paired:
        write_json(out/'MATCHED_TASKS.json', matched)
    out.mkdir(parents=True, exist_ok=True)
    (out/'STUDENT_SYSTEM_caption_v6.txt').write_text(proto.SYSTEM + '\n')
    for name, (rows, ledger) in built.items():
        stats = summary(rows, ledger)
        if paired:
            stats['matched'] = summary([r for r in rows if r['audit_id'] in set(matched)], ledger)['all']
        (out/name).mkdir(parents=True, exist_ok=True)
        (out/name/'DATABASE.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        write_json(out/name/'LEDGER_4040.json', ledger)
        write_json(out/name/'SUMMARY.json', stats)
        report[name] = {k: stats[k] for k in ('in_library', 'missing', 'never_attempted', 'accepted_by')} | dict(
            mean_steps=stats['all']['mean_steps'], median_steps=stats['all']['median_steps'],
            by_generator={g: t['tasks'] for g, t in stats['by_generator'].items()})
        if repo is not None:
            train = sharegpt(rows, name, repo)
            write_json(repo/'data'/f'caption_sft_v4c_{name}_train.json', train, compact=True)
            if paired:
                write_json(repo/'data'/f'caption_sft_v4c_{name}_matched.json', [t for t in train if t['id'] in set(matched)], compact=True)
            meta = repo/'data/interp_v4c'/name
            meta.mkdir(parents=True, exist_ok=True)
            for f in ('LEDGER_4040.json', 'SUMMARY.json'):
                shutil.copyfile(out/name/f, meta/f)
            (meta/'DATABASE.jsonl').write_text(''.join(json.dumps(dict(r, images=t['images']), ensure_ascii=False) + '\n'
                                                       for r, t in zip(rows, train)))
    if repo is not None:
        if paired:
            shutil.copyfile(out/'MATCHED_TASKS.json', repo/'data/interp_v4c/MATCHED_TASKS.json')
        shutil.copyfile(out/'STUDENT_SYSTEM_caption_v6.txt', repo/'data/interp_v4c/STUDENT_SYSTEM_caption_v6.txt')
    write_json(out/'REPORT.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--repo', type=Path, help='export training files and images into this data repository')
    parser.add_argument('--allow-running', action='store_true', help='dry run while waves are still running')
    parser.add_argument('--libraries', nargs='+', choices=sorted(LIBRARIES), default=sorted(LIBRARIES, reverse=True),
                        help='which libraries to assemble (the control library was stopped unfinished on 2026-10-13)')
    args = parser.parse_args()
    print(json.dumps(build(args.out.resolve(), args.repo.resolve() if args.repo else None, args.allow_running, args.libraries), ensure_ascii=False))
