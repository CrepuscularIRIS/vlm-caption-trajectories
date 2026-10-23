"""Read-only point checks. Usage: python3 audit_stage0.py RAW_DIR [OUTPUT_JSON].
Raw fork data stay private. Does not bootstrap, load models, or write source inputs.
"""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('stage0', root / 'logs/dl/stage0/stage0.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
raw = Path(sys.argv[1])
audit = [s for s in json.loads((raw / 'audit_states.json').read_text())['states'] if s['usable'] and len(s['forks']) == 8]
ref = {s['uid']: s for s in json.loads((raw / 'score_ref.json').read_text())['scores']}
forks = [s['forks'] for s in audit]
d = [m.gap(s, range(4)) for s in forks]
z = [ref[s['uid']]['z'] for s in audit]
assert all(collections.Counter((f['action'], f['k']) for f in s) == collections.Counter((a, k) for a in ('answer', 'grounding') for k in range(4)) for s in forks)
threshold, gain = m.best_threshold(z, d)
positive = [{'uid':s['uid'], 'D':v, 'z':zz} for s,v,zz in zip(audit,d,z) if v > 0]
out = {'scope':'point metrics only; no new sampling or bootstrap',
       'input_sha256':{name:hashlib.sha256((raw/name).read_bytes()).hexdigest() for name in ('audit_states.json','score_ref.json')},
       'states':len(audit), 'complete_unique_action_k':True,
       'decompose':m.decompose(forks), 'positive_states':positive,
       'positive_high_z_gt14':sum(s['z']>14 for s in positive),
       'all_high_z_gt14':[{'uid':s['uid'],'D':v,'z':zz} for s,v,zz in zip(audit,d,z) if zz>14],
       'crop_branch_counts':dict(sorted(collections.Counter(f['n_crops'] for s in forks for f in s if f['action']=='grounding').items())),
       'threshold':threshold,'same_table_oracle_minus_threshold':sum(max(x,0) for x in d)/len(d)-gain}
rendered = json.dumps(out, indent=2)
if len(sys.argv)>2:
    destination=Path(sys.argv[2]).resolve()
    assert not destination.is_relative_to(raw.resolve()), 'Do not write into raw input directory'
    destination.write_text(rendered+'\n')
print(rendered)
