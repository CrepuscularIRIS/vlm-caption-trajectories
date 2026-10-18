"""Funnel v1 item pool: all corpus_v2 train items not used by earlier runs, in a balanced order.

Order = round-robin over (bench, material) cells, each cell in salted-hash order, so any prefix of the
pool is source-balanced; MME-RealWorld (two thirds of train) fills the tail once smaller cells run out.
Per-bench caps are applied at selection time, not here. Gold stays in the pool file but never reaches
the generator (funnel_gen passes an allowlist of public fields).
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / 'obs/benchmarks/corpus_v2/items.jsonl'
SALT = 'funnel_v1_20261002'
USED_ITEM_FILES = ['obs/benchmarks/smoke_v51_20261001', 'obs/benchmarks/cover_v51_20261001',
                   'obs/benchmarks/need_pilot_20261002', 'obs/benchmarks/batch1_20261002',
                   'obs/runs/binding_unit1_20261002']
FEWSHOT = ROOT / 'obs/notes/traj_replica_20260929/tiered_binding_20261002/fewshot'


def lines(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def used_groups() -> tuple[set[str], list[str]]:
    groups, files = set(), []
    for d in USED_ITEM_FILES:
        for p in sorted((ROOT / d).glob('items*.jsonl')):
            groups.update(i['group_id'] for i in lines(p))
            files.append(str(p.relative_to(ROOT)))
    for p in FEWSHOT.glob('*__*.json'):
        groups.add(json.loads(p.read_text())['group_id'])
    return groups, files


def balanced_order(items: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for it in items:
        cells[(it['bench'], it['material'])].append(it)
    for q in cells.values():
        q.sort(key=lambda i: hashlib.sha256((SALT + i['item_id']).encode()).hexdigest())
    order, keys = [], sorted(cells)
    while any(cells[k] for k in keys):
        for k in keys:
            if cells[k]:
                order.append(cells[k].pop(0))
    return order


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    groups, files = used_groups()
    train = [i for i in lines(CORPUS) if i['split'] == 'train']
    pool = balanced_order([i for i in train if i['group_id'] not in groups])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'pool.jsonl').write_text(''.join(json.dumps(i, ensure_ascii=False) + '\n' for i in pool))
    summary = dict(salt=SALT, corpus_sha256=hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
                   train_items=len(train), excluded_groups=len(groups), excluded_from=files, pool_items=len(pool),
                   by_bench=dict(Counter(i['bench'] for i in pool)),
                   by_material=dict(Counter(i['material'] for i in pool)))
    (a.out / 'POOL.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
