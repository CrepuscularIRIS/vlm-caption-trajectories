"""Construction v4 under the first library's conditions ("v4c"): new design kept, other variables held.

2026-10-13 user decision: redo everything. Keep the v4 design (caption v6 skill card the student knows in
advance, three-line captions, forced misses on the generator's own crop requests with those turns at loss 0,
teacher-side construction condition), but hold the variables of the first SFT library: the same 4040 tasks,
the 2 MP view delivery and a crop budget of 8. Construction v4 had delivered views with the longer side at
672/448/336 and budgets of 8/12/16, which cost answer accuracy and made the two libraries differ in more
than the interpolation.

The frozen v4 constructor already takes delivery scale and budget from the variant file; this wrapper only
records the decision, freezes itself with the campaign and makes launched workers start here. Generator:
GPT-6 Astra. Earlier campaigns are read only.
"""
import argparse
import json
from pathlib import Path

import interp_v4r2_campaign_20261013 as v4r2

v4, batch, base, campaign, plan = v4r2.v4, v4r2.batch, v4r2.base, v4r2.campaign, v4r2.plan
THIS_FILE = Path(__file__).resolve()
PARENT = v4.ROOT/'obs/runs/interp_v4c_astra_20261013'
VARIANT = v4.ROOT/'obs/runs/interp_v4c_variants_20261013/ctrl_2mp_b8.json'
AUTHORIZATION = ('2026-10-13 user: redo all trajectories; keep the new design (caption v6 skill card, forced interpolation) '
                 'but hold the other variables of the first library (2 MP view delivery, budget 8, same tasks). '
                 'GPT-6 Astra. No K3 or Gemini. Own ledger.')

v4.AUTHORIZATION = base.AUTHORIZATION = AUTHORIZATION
v4.WRAPPERS = (*v4.WRAPPERS, THIS_FILE)
base.__file__ = str(THIS_FILE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'tick', 'run'])
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--wave', type=Path)
    parser.add_argument('--limit', type=int, help='build only the first N tasks of the hash order (pilot)')
    parser.add_argument('--exclude', type=Path, action='append', default=[], help='v4c campaign whose tasks are already issued')
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
