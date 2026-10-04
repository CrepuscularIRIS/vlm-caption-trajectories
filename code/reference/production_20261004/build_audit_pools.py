"""P3 for a production audit run: pools + a 5 % stratified human spot-check of the auto-passed (clean) pool.

Construction mode is single_teacher_multi_judge (one Luna trajectory, process review by Sonnet, blind facts by
Gemini or Grok; each sidecar records its reviewers). Pool rules are shared with build_unit_casebook.pool_of.
Not-yet-audited rows (PENDING_AUDIT_BUDGET, AUDIT_ERROR, TRANSPORT_INVALID, SKIPPED) are reported as 'pending'.
Spot-check: deterministic hash order inside each (tag, material) stratum of the clean pool, ceil(5 %), at least 2.
Outputs in the audit dir: POOLS.json, HUMAN_QUEUE.jsonl (disputed + review_format), SPOTCHECK_TEMPLATE.json,
SPOTCHECK_CASEBOOK.html.
"""
import argparse
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from build_unit_casebook import CSS, card, pool_of

PENDING = {'PENDING_AUDIT_BUDGET', 'AUDIT_ERROR', 'TRANSPORT_INVALID', 'SKIPPED'}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--rate', type=float, default=0.05)
    a = p.parse_args()
    rows = []
    for f in sorted((a.audit / 'audit').glob('*.json')):
        s = json.loads(f.read_text())
        r = json.loads(Path(s['result_path']).read_text())
        pool = 'pending' if s['decision'] in PENDING else pool_of(s, r)
        rows.append(dict(key=f.stem, pool=pool, tag=s['manifest_row'].get('tag'), material=r['item']['material'],
                         bench=r['item']['bench'], item_id=s['item_id'], group_id=r['item']['group_id'],
                         decision=s['decision'], reasons=s.get('reasons'), process_reviewer=s.get('process_reviewer', 'claude-sonnet-5-5'),
                         blind_reader=s.get('blind_reader', 'gemini-3.8-flash-low'), result_path=s['result_path'],
                         construction_mode='single_teacher_multi_judge', side=s, result=r))
    pools = Counter(x['pool'] for x in rows)
    table = defaultdict(Counter)
    for x in rows:
        table[f"{x['tag']}/{x['material']}"][x['pool']] += 1
    summary = dict(construction_mode='single_teacher_multi_judge', rows=len(rows), pools=dict(pools),
                   clean_groups=len({x['group_id'] for x in rows if x['pool'] == 'clean_candidate'}),
                   by_tag_material={k: dict(v) for k, v in sorted(table.items())},
                   readers=dict(Counter(x['blind_reader'] for x in rows if x['pool'] != 'pending')))
    (a.audit / 'POOLS.json').write_text(json.dumps(dict(summary, rows={x['key']: x['pool'] for x in rows}), indent=1))
    human = [dict(key=x['key'], pool=x['pool'], item_id=x['item_id'], reasons=x['reasons']) for x in rows
             if x['pool'] in ('disputed', 'review_format')]
    (a.audit / 'HUMAN_QUEUE.jsonl').write_text(''.join(json.dumps(h) + '\n' for h in human))
    strata = defaultdict(list)
    for x in rows:
        if x['pool'] == 'clean_candidate':
            strata[(x['tag'], x['material'])].append(x)
    picks = []
    for key, xs in sorted(strata.items()):
        xs.sort(key=lambda x: hashlib.sha256(('spot' + x['key']).encode()).hexdigest())
        picks += xs[:min(len(xs), max(2, math.ceil(a.rate * len(xs))))]
    (a.audit / 'SPOTCHECK_TEMPLATE.json').write_text(json.dumps(
        {x['key']: dict(verdict='PENDING', stratum=f"{x['tag']}/{x['material']}", reviewer='', note='') for x in picks},
        indent=1, ensure_ascii=False))
    toc = ''.join(f"<li><a href='#{html.escape(x['key'])}'>{html.escape(x['key'])}</a> — {x['tag']}/{x['material']}</li>" for x in picks)
    body = '\n'.join(card(x['key'], x['side'], x['result'], x['pool']) for x in picks)
    (a.audit / 'SPOTCHECK_CASEBOOK.html').write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Luna spot-check</title><style>{CSS}</style></head><body><h1>Luna production audit — {len(picks)}-trajectory spot-check</h1>"
        f"<p>Stratified {a.rate:.0%} of the auto-passed pool ({pools.get('clean_candidate', 0)} trajectories). Mark each KEEP / WEAK / BAD in "
        f"SPOTCHECK_TEMPLATE.json. Pools: {dict(pools)}.</p><ol>{toc}</ol>{body}</body></html>")
    print(json.dumps(dict(summary, spotcheck=len(picks), human_queue=len(human)), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
