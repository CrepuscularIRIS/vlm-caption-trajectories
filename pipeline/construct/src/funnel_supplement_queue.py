"""Selective cross-generation queue (MENTOR_ALIGNED_FUNNEL_PLAN_20261003 §L2): which items get an independent
second-family trajectory, in priority order. Luna stays the main generator; GLM (and K3/K2.8 when quota allows)
generate only for queued items. Items that already have a result from a non-OpenAI family are skipped.

Priority (highest first):
  1 core      Luna correct crop trajectories that passed CPU gates (binding / active observation / keep material),
              audit AUTO_PASS or DISPUTED before not-yet-audited, rare signals (HOLD, REVISE, fallback candidate) first
  2 control   a 10 % hash sample of Luna correct direct answers (so consensus is not only formed on hard items)
  3 unsolved  Luna wrong or UNCLEAR on non-academic material (evidence may be obtainable)
Writes <root>/supplement_queue.jsonl ({item_id, reason}) and SUPPLEMENT_QUEUE.json (counts). Read-only otherwise.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import funnel_select as fs

OPENAI = {'gpt-6-luna', 'gpt-6.1-sol', 'gpt-5.6-sol', 'gpt-6-sol', 'gpt-6-astra'}
AUDIT_RANK = {'AUTO_PASS_PENDING_MANUAL': 0, 'DISPUTED_PENDING_MANUAL': 1, None: 2}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True, help='audit run dir with audit/*.json sidecars')
    a = p.parse_args()
    latest = {}
    for line in (a.root / f'select_{fs.SELECT_VERSION}' / 'traj_cache.jsonl').read_text().splitlines():
        r = json.loads(line)
        if Path(r['path']).exists():
            latest[r['path']] = r['f']
    by_item = {}
    for f in latest.values():
        by_item.setdefault(f['item_id'], []).append(f)
    decision = {}
    for s in (a.audit / 'audit').glob('*.json'):
        d = json.loads(s.read_text())
        decision[d['item_id']] = d.get('decision')
    rows = []
    for iid, fs_ in by_item.items():
        if any(f['model'] not in OPENAI for f in fs_):
            continue  # already has an independent non-OpenAI trajectory
        luna = next((f for f in fs_ if f['model'] == 'gpt-6-luna'), None)
        if luna is None:
            continue
        ok = luna['term'] == 'answer' and luna['score'] == 1 and not fs.gate(luna)
        if ok and luna['crops'] > 0:
            if decision.get(iid) == 'REVIEW_REJECTED':
                continue  # the Luna path itself failed review: a second path would not rescue it as Luna material
            rare = luna['holds'] + luna['revises'] + luna['fallback_candidates'] > 0
            key = (1, AUDIT_RANK.get(decision.get(iid), 2), not rare, iid)
            rows.append((key, dict(item_id=iid, reason='core_crop', audit=decision.get(iid), rare=rare)))
        elif ok and int(hashlib.sha256(('ctrl' + iid).encode()).hexdigest(), 16) % 10 == 0:
            rows.append(((2, 0, False, iid), dict(item_id=iid, reason='direct_control')))
        elif not ok and luna['term'] in ('answer', 'abstain') and luna['material'] != 'academic':
            rows.append(((3, 0, False, iid), dict(item_id=iid, reason='luna_unsolved', material=luna['material'])))
    rows.sort(key=lambda x: x[0])
    out = [r for _, r in rows]
    (a.root / 'supplement_queue.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in out))
    summary = dict(total=len(out), by_reason=dict(Counter(r['reason'] for r in out)),
                   core_by_audit=dict(Counter(str(r.get('audit')) for r in out if r['reason'] == 'core_crop')))
    (a.root / 'SUPPLEMENT_QUEUE.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
