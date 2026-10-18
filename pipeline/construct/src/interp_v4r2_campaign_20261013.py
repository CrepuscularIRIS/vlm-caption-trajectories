"""Construction v4, revision 2: keep the first complete turn when a reply carries trailing text.

Pilot 1 (40 tasks, GPT-6 Astra) lost 4 episodes to wire artefacts after a well-formed first turn:
stray tokens (" handoff to=all hresult?") or an imagined second action ("<answer>UNCLEAR</answer>"
right after the crop request). The environment executes one action per turn, so nothing after the
first action can have influenced it. The remainder is dropped from the episode and the full wire
text stays in the receipt, as v3r2 already did for imagined follow-on turns.

Everything else is the frozen v4 constructor; variants stay data.
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import interp_v4_campaign_20261013 as v4

r2, batch, base, campaign, plan = v4.r2, v4.batch, v4.base, v4.campaign, v4.plan
THIS_FILE = Path(__file__).resolve()
PARENT = v4.ROOT/'obs/runs/interp_v4_astra_20261013'


def leading_turn(text: str) -> Optional[str]:
    """The first complete turn when anything at all follows it; else None."""
    match = r2.FIRST_TURN.match(text)
    if not match or not text[match.end():].strip():
        return None
    return text[:match.end()]


class V4Transport(r2.V3Transport):
    def api_call(self, msgs, max_tokens, temperature, top_p):
        reply = super().api_call(msgs, max_tokens, temperature, top_p)
        single = leading_turn(reply.get('content') or '')
        if single is None:
            return reply
        wire = reply.get('original_wire_content', reply['content'])
        return dict(reply, content=single, original_wire_content=wire,
                    wire_normalization=dict(kind='trailing_text_dropped', changed_text=False,
                                            dropped_chars=len(reply['content']) - len(single),
                                            source_sha256=hashlib.sha256(wire.encode()).hexdigest()))


def install_run() -> None:
    base.set_route()
    batch.Transport = V4Transport
    batch.generate = v4.generate


v4.WRAPPERS = (*v4.WRAPPERS, THIS_FILE)
base.__file__ = str(THIS_FILE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'tick', 'run'])
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--wave', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--exclude', type=Path, action='append', default=[])
    parser.add_argument('--variant-file', type=Path)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    target = args.parent.resolve()
    if args.action == 'build':
        variant = dict(plan.DEFAULT_VARIANT, **(batch.read(args.variant_file) if args.variant_file else {}))
        print(json.dumps(v4.build(target, v4.SOURCE, args.limit, [e.resolve() for e in args.exclude], variant)))
    elif args.action == 'tick':
        print(json.dumps(base.tick(target, args.launch)))
    else:
        install_run()
        campaign.run_wave(target, args.wave.resolve(), 'sol')
