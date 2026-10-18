"""Keep corpus_v2's sealed splits in an extended build (e.g. corpus_v2_mmecomplete_20261003).

Replacing Lite placeholders by full-resolution images changes image hashes, so build_corpus_v2 re-groups those
items and may draw a new split from the salted hash. corpus_v2 items are sealed (the funnel already generated on
its train split), so here:
- an old item is matched by (item_id, image_sha256), else by item_id/lite alias when the id is unique
  (VisReason two-image items repeat their item_id, one row per image);
- every new group that contains old items inherits their old split; if those old splits disagree, the group keeps
  the old per-item splits and its fresh items are quarantined;
- fresh items in untouched groups keep the build's split.
Rewrites items.jsonl / groups.jsonl of the NEW corpus only (originals kept as *.pre_reconcile) and writes
RECONCILE.json. Re-run verify_corpus_extension.py afterwards.
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def lines(p: Path) -> list[dict]:
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def old_split_for(new_items: list[dict], old_items: list[dict]) -> dict[int, str]:
    by_key = {(o['item_id'], o['image_sha256']): o for o in old_items}
    id_count = Counter(o['item_id'] for o in old_items)
    by_id = {o['item_id']: o for o in old_items if id_count[o['item_id']] == 1}
    out = {}
    for k, n in enumerate(new_items):
        o = by_key.get((n['item_id'], n.get('image_sha256'))) or by_id.get(n['item_id']) or by_id.get(n.get('lite_id'))
        if o is not None:
            out[k] = o['split']
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    a = p.parse_args()
    old, new = lines(a.old / 'items.jsonl'), lines(a.new / 'items.jsonl')
    groups = {g['group_id']: g for g in lines(a.new / 'groups.jsonl')}
    sealed = old_split_for(new, old)
    by_group = defaultdict(list)
    for k, n in enumerate(new):
        by_group[n.get('group_id')].append(k)
    changed_groups, conflict_groups, quarantined, item_fixes = 0, 0, 0, 0
    for gid, ks in by_group.items():
        olds = {sealed[k] for k in ks if k in sealed}
        if not olds or gid is None:
            continue
        if len(olds) == 1:
            split = olds.pop()
            if groups[gid]['split'] != split:
                groups[gid]['split'] = split
                changed_groups += 1
            for k in ks:
                item_fixes += new[k]['split'] != split
                new[k]['split'] = split
        else:  # old items of one new group sat in different splits: keep them, quarantine the fresh ones
            conflict_groups += 1
            for k in ks:
                if k in sealed:
                    item_fixes += new[k]['split'] != sealed[k]
                    new[k]['split'] = sealed[k]
                else:
                    new[k]['split'] = 'quarantine'
                    quarantined += 1
    for f in ('items.jsonl', 'groups.jsonl'):
        backup = a.new / (f + '.pre_reconcile')
        if not backup.exists():
            shutil.copy(a.new / f, backup)
    (a.new / 'items.jsonl').write_text(''.join(json.dumps(n, ensure_ascii=False) + '\n' for n in new))
    (a.new / 'groups.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in groups.values()))
    rep = dict(old_items_matched=len(sealed), groups_split_restored=changed_groups, conflict_groups=conflict_groups,
               fresh_items_quarantined=quarantined, item_split_fixes=item_fixes)
    (a.new / 'RECONCILE.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
