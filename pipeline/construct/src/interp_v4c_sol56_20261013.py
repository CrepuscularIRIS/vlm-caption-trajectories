"""Construction v4c with GPT-5.6 Sol as the generator (pilot comparison).

2026-10-13 user decision: after GPT-6 Astra and GPT-6.1 Sol gave the same step counts on the pilot tasks,
try GPT-5.6 Sol as well. Same v4c construction and variant (caption v6, 2 MP delivery, budget 8, forced
misses); own ledger. Other campaigns are read only.
"""
import argparse
import json
from pathlib import Path

import interp_v4r2_campaign_20261013 as v4r2

v4, batch, base, campaign, plan = v4r2.v4, v4r2.batch, v4r2.base, v4r2.campaign, v4r2.plan
MODEL = 'gpt-5.6-sol'
THIS_FILE = Path(__file__).resolve()
PARENT = v4.ROOT/'obs/runs/interp_v4c_sol56_20261013'
VARIANT = v4.ROOT/'obs/runs/interp_v4c_variants_20261013/ctrl_2mp_b8.json'
AUTHORIZATION = ('2026-10-13 user: also try GPT-5.6 Sol as generator under the v4c construction '
                 '(first-library image delivery and budget, new design kept). No K3 or Gemini. Own ledger.')

v4.install(MODEL)
v4.MODEL = MODEL
v4.AUTHORIZATION = base.AUTHORIZATION = AUTHORIZATION
v4.WRAPPERS = (*v4.WRAPPERS, THIS_FILE)
base.__file__ = str(THIS_FILE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'tick', 'run'])
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--wave', type=Path)
    parser.add_argument('--limit', type=int, help='build only the first N tasks of the hash order (pilot)')
    parser.add_argument('--exclude', type=Path, action='append', default=[], help='campaign whose tasks this model already ran')
    parser.add_argument('--variant-file', type=Path, default=VARIANT)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    target = args.parent.resolve()
    if args.action == 'build':
        variant = dict(plan.DEFAULT_VARIANT, **batch.read(args.variant_file.resolve()))
        print(json.dumps(v4.build(target, v4.SOURCE, args.limit, [e.resolve() for e in args.exclude], variant)))
    elif args.action == 'tick':
        print(json.dumps(base.tick(target, args.launch)))
    else:
        v4r2.install_run()
        campaign.run_wave(target, args.wave.resolve(), 'sol')
