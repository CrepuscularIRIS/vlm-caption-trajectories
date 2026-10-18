"""Pro-only Sol continuation: adds bounded client backoff for no-output overload errors.

Builds on ``interp_sol_pro_retry_dedup_20261012`` (retry-enabled owned proxy on
8323 plus exact-duplicate wire collapse) without editing it. The proxy retries
immediately, but upstream ``server_is_overloaded`` comes in bursts, so ~4% of
client calls still failed and each failure ended an episode. Here a call that
returned no model output with that specific error is resent after a pause.
Every failed attempt stays in the receipt under ``transport_retries``.

Model outputs are never re-rolled: anything with content, a quota/auth/request
error, or any other failure is returned unchanged on the first attempt.
"""
import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import interp_sol_pro_retry_dedup_20261012 as dedup
import interp_sol_full_campaign as campaign
import interp_sol_full_4040 as batch
from tiered_transport import write_json

base = dedup.base
DEDUP_FILE = dedup.THIS_FILE
THIS_FILE = Path(__file__).resolve()
BACKOFF_SECONDS: Sequence[float] = (8, 20, 45)
RETRY_STATUS = (502, 503)
RETRY_CODE = 'server_is_overloaded'

base.AUTHORIZATION = dedup.base.AUTHORIZATION + ' Bounded client backoff for no-output server_is_overloaded.'
# launch() re-enters the file named here.
base.__file__ = str(THIS_FILE)


def is_overloaded(reply: dict) -> bool:
    """True only for a reply with no model output whose every error is the overload error."""
    errors = reply.get('errors') or []
    if not errors or reply.get('content') or reply.get('reasoning_content'):
        return False
    return all(e.get('status') in RETRY_STATUS and (e.get('detail') or {}).get('code') == RETRY_CODE for e in errors)


class BackoffTransport(dedup.DedupTransport):
    def api_call(self, msgs, max_tokens, temperature, top_p):
        failed = []
        for wait in (*BACKOFF_SECONDS, None):
            reply = super().api_call(msgs, max_tokens, temperature, top_p)
            if wait is None or not is_overloaded(reply):
                break
            failed.append(dict(errors=reply['errors'], waited_seconds=wait))
            time.sleep(wait)
        return dict(reply, transport_retries=failed) if failed else reply


def install_run() -> None:
    dedup.install_run()
    batch.Transport = BackoffTransport


def prepare_primary_config(parent: Path, loc: Path) -> set:
    """Point a never-issued primary wave at this route and freeze the wrapper chain."""
    ids = _dedup_prepare(parent, loc)
    cfg = batch.read(loc/'config.json')
    cfg['frozen_files'][str(DEDUP_FILE)] = batch.old.sha(DEDUP_FILE)
    cfg['client_transport_retry'] = dict(status=list(RETRY_STATUS), code=RETRY_CODE, backoff_seconds=list(BACKOFF_SECONDS),
                                         only_when_no_model_output=True, failed_attempts_kept_in_receipt='transport_retries',
                                         ledger_counts_one_client_request_per_label=True)
    write_json(loc/'config.json', cfg)
    (loc/'CONFIG.sha256').write_text(batch.old.sha(loc/'config.json')+'\n')
    branch = dict(cfg, frozen_files=dict(cfg['frozen_files'], **{str(loc/'config.json'): batch.old.sha(loc/'config.json')}))
    write_json(loc/'sol/config.json', branch)
    (loc/'sol/CONFIG.sha256').write_text(batch.old.sha(loc/'sol/config.json')+'\n')
    batch.verify(loc, 'sol')
    return ids


_dedup_prepare = dedup.prepare_primary_config
base.prepare_primary_config = prepare_primary_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['tick', 'run'])
    parser.add_argument('--parent', type=Path, default=campaign.PARENT)
    parser.add_argument('--wave', type=Path)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    parent = args.parent.resolve()
    if args.action == 'tick':
        print(json.dumps(base.tick(parent, args.launch)))
    else:
        install_run()
        campaign.run_wave(parent, args.wave.resolve(), 'sol')
