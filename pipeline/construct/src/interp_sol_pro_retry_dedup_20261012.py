"""Pro-only Sol continuation: retry-enabled owned proxy plus exact-duplicate wire collapse.

Wraps the frozen ``interp_sol_pro_continue_20261012`` controller without editing
it. Two differences from that controller, both transport-level:

* route: owned CLIProxyAPI on port 8323 that retries transient upstream failures;
* wire: when the returned content is one protocol turn repeated twice verbatim
  (observed on ~30% of observe-first first turns through the Codex route), keep a
  single copy. The original wire text and its hash stay in the receipt.

Prompts, constructor, ledger, one-episode-per-task rule and caps are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import interp_sol_pro_continue_20261012 as base
import interp_sol_full_campaign as campaign
import interp_sol_full_4040 as batch
from tiered_transport import write_json

BASE_FILE = Path(base.__file__).resolve()
THIS_FILE = Path(__file__).resolve()
MIN_TURN_CHARS = 40

base.PORT = 8323
base.ENDPOINT = f'http://127.0.0.1:{base.PORT}'
base.PROXY_RECORD = 'SOL_RETRY_PRO_PROXY_LAUNCH.json'
base.AUTHORIZATION = ('2026-10-12 user in new chat: continue trajectory interpolation for all 4040 with '
                      'ChatGPT OAuth GPT5.6Sol through CLIProxyAPI; build intermediate steps for single-step '
                      'items via prompt. Pro-only Sol48/90RPM low; proxy-level transient retry; exact duplicate '
                      'wire text collapsed; previous episodes preserved and never reissued.')
# launch() re-enters the file named here; it must be this wrapper so the worker
# also uses the retry-enabled route and the wire collapse.
base.__file__ = str(THIS_FILE)


def collapse_exact_duplicate(text: str) -> Optional[str]:
    """Return one copy when ``text`` is a single turn repeated twice verbatim, else None."""
    half, odd = divmod(len(text), 2)
    if odd or half < MIN_TURN_CHARS or text[:half] != text[half:]:
        return None
    return text[:half]


class DedupTransport(base.ObserveFirstTransport):
    def api_call(self, msgs, max_tokens, temperature, top_p):
        reply = super().api_call(msgs, max_tokens, temperature, top_p)
        single = collapse_exact_duplicate(reply.get('content') or '')
        if single is None:
            return reply
        wire = reply['content']
        return dict(reply, content=single, original_wire_content=wire,
                    wire_normalization=dict(kind='exact_duplicate_collapsed', copies=2, changed_text=False,
                                            source_sha256=hashlib.sha256(wire.encode()).hexdigest()))


def install_run() -> None:
    base.install_run()
    batch.Transport = DedupTransport


def prepare_primary_config(parent: Path, loc: Path) -> set:
    """Point a never-issued primary wave at the retry proxy and freeze both sources."""
    ids = _base_prepare(parent, loc)
    cfg = batch.read(loc/'config.json')
    cfg['frozen_files'][str(BASE_FILE)] = batch.old.sha(BASE_FILE)
    cfg['provider_route'] = 'native Pro-only Codex GPT5.6Sol via retry-enabled owned proxy; no Cursor/xAI'
    cfg['proxy_retry'] = dict(request_retry=3, max_retry_interval_seconds=30, transient_error_cooldown='disabled',
                              upstream_attempts_observable=False)
    cfg['wire_normalization'] = dict(kind='exact_duplicate_collapsed', min_turn_chars=MIN_TURN_CHARS,
                                     original_wire_content_kept_in_receipt=True)
    write_json(loc/'config.json', cfg)
    (loc/'CONFIG.sha256').write_text(batch.old.sha(loc/'config.json')+'\n')
    branch = dict(cfg, frozen_files=dict(cfg['frozen_files'], **{str(loc/'config.json'): batch.old.sha(loc/'config.json')}))
    write_json(loc/'sol/config.json', branch)
    (loc/'sol/CONFIG.sha256').write_text(batch.old.sha(loc/'sol/config.json')+'\n')
    batch.verify(loc, 'sol')
    return ids


_base_prepare = base.prepare_primary_config
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
