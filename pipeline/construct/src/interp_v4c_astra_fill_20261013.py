"""v4c, third generator: GPT-6 Astra on the tasks GPT-5.6 Sol and GPT-6.1 Sol left without a correct trajectory.

2026-10-13 user decision: GPT-5.6 Sol runs everything; what it does not solve goes once to GPT-6.1 Sol and
then once to GPT-6 Astra. Same v4c construction and variant. A task this model already ran in a v4c
campaign is not run again; that saved episode is its one attempt. Correctness is decided offline by
``interp_v4c_select_20261013``. Earlier campaigns are read only.
"""
import argparse
import json
from pathlib import Path

import interp_v4c_astra_20261013 as first
import interp_v4c_select_20261013 as select
import interp_v4c_pace_20261013 as pace

v4, batch, base, campaign, v4r2 = first.v4, first.batch, first.base, first.campaign, first.v4r2
THIS_FILE = Path(__file__).resolve()
RUNS = v4.ROOT/'obs/runs'
SOLVED_BY = [RUNS/'interp_v4c_sol56_pilot1_20261013', RUNS/'interp_v4c_sol56_20261013', RUNS/'interp_v4c_sol56_laneA_20261013',
             RUNS/'interp_v4c_sol56_laneB_20261013', RUNS/'interp_v4c_sol56_rest_20261013', RUNS/'interp_v4c_sol61_pilot1_20261013', RUNS/'interp_v4c_sol61_fill_20261013']
OWN = [RUNS/'interp_v4c_pilot1_20261013', RUNS/'interp_v4c_astra_20261013']
PARENT = RUNS/'interp_v4c_astra_fill_20261013'
AUTHORIZATION = ('2026-10-13 user: GPT-5.6 Sol runs all tasks; tasks without a correct trajectory go once to GPT-6.1 Sol, '
                 'then once to GPT-6 Astra. This campaign is the GPT-6 Astra pass. No K3 or Gemini. Own ledger.')

v4.AUTHORIZATION = base.AUTHORIZATION = AUTHORIZATION
v4.WRAPPERS = (*v4.WRAPPERS, Path(select.__file__).resolve(), THIS_FILE)
pace.apply()
base.__file__ = str(THIS_FILE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'tick', 'run'])
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--wave', type=Path)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    target = args.parent.resolve()
    if args.action == 'build':
        print(json.dumps(select.build_fill(v4, target, SOLVED_BY, OWN, first.VARIANT)))
    elif args.action == 'tick':
        print(json.dumps(base.tick(target, args.launch)))
    else:
        v4r2.install_run()
        campaign.run_wave(target, args.wave.resolve(), 'sol')
