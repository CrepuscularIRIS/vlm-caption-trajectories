"""Construction v3 campaign: intent-miss interpolation with a Sol -> K3 -> Gemini funnel.

User decision 2026-10-12: rebuild the interpolation along the full-process audit
(pick conditions that are really unresolved, vary the insertion time, allow a
reasonable long search, keep short paths and timely stopping) and continue with
the funnel design. Planning and geometry live in ``interp_v3_plan_20261012``.

This is a new construction version in its own campaign directory with its own
ledger. r2 adds: variants as data, a primary model other than Sol for hybrid pilots, and
keeping only the first turn when a reply runs on into an imagined next turn. Earlier campaigns, their results, GT, splits and frozen sources are read
only. One episode per model and task inside this version; a valid wrong answer
is never re-rolled on the same model, it moves down the funnel.
"""
import argparse
import collections
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time
from typing import Dict, Iterable, List, Optional

from PIL import Image

import interp_sol_pro_retry_backoff_20261012 as backoff
import interp_sol_full_campaign as campaign
import interp_sol_full_4040 as batch
import interp_v3r2_plan_20261012 as plan
from tiered_transport import ROOT, write_json

base = backoff.base
dedup = backoff.dedup
old = batch.old
proto = batch.proto
THIS_FILE = Path(__file__).resolve()
PLAN_FILE = Path(plan.__file__).resolve()
CAMPAIGN_FILE = Path(campaign.__file__).resolve()
SOURCE = campaign.PARENT
PARENT = ROOT/'obs/runs/interp_v3r2_intent_miss_20261012'
QUEUES = ROOT/'reports/interp_sol_full_4040_20261011/full_process_audit_20261012/QUEUES.json'
KEEP_EXISTING = ('RETAIN_REVIEWED_PATH_WITH_BEHAVIOR_LABEL', 'ORDINARY_SOLVE_RETAIN_CANDIDATE_NOT_FORCE_INTERPOLATION')
SOL = 'gpt-5.6-sol'
BUDGET = 8
MAX_STEPS = 9
DEADLINE_DAYS = 5
WRAPPERS = (THIS_FILE, PLAN_FILE, backoff.THIS_FILE, dedup.THIS_FILE, dedup.BASE_FILE, CAMPAIGN_FILE)
AUTHORIZATION = ('2026-10-12 user: continue along the full-process audit and the funnel design; goal is the '
                 'mentor requirement; produce the 4040 trajectory interpolation; redesign the inserted caption. '
                 'New construction version, own ledger; earlier campaigns read only.')

FIRST_TURN = re.compile(r'\s*<(caption|think)>.*?</\1>\s*<(grounding|answer)>.*?</\2>', re.S)


def first_turn(text: str) -> Optional[str]:
    """One complete turn when the reply continues into a further, imagined turn; else None.

    The environment executes one action per turn. Text after it claims views that were never
    delivered; it cannot have influenced the turn before it and is dropped from the episode
    (the full wire text stays in the receipt).
    """
    match = FIRST_TURN.match(text)
    if not match:
        return None
    rest = text[match.end():].lstrip()
    return text[:match.end()] if rest.startswith(('<caption>', '<think>')) else None

base.AUTHORIZATION = AUTHORIZATION
# launch() re-enters the file named here.
base.__file__ = str(THIS_FILE)


def with_condition(messages: List[dict], text: str) -> List[dict]:
    """Append the teacher-only construction condition to the system turn of a request."""
    if not text:
        return messages
    result = copy.deepcopy(messages)
    system = result[0]
    if system.get('role') != 'system' or not isinstance(system.get('content'), list):
        raise ValueError('Expected a text-block system protocol')
    system['content'].append(dict(type='text', text=text))
    return result


class V3Transport(batch.Transport):
    """Frozen transport plus the two wire-level fixes validated on 2026-10-12."""

    def api_call(self, msgs, max_tokens, temperature, top_p):
        failed = []
        for wait in (*backoff.BACKOFF_SECONDS, None):
            reply = super().api_call(msgs, max_tokens, temperature, top_p)
            if wait is None or not backoff.is_overloaded(reply):
                break
            failed.append(dict(errors=reply['errors'], waited_seconds=wait))
            time.sleep(wait)
        single = dedup.collapse_exact_duplicate(reply.get('content') or '')
        if single is not None:
            wire = reply['content']
            reply = dict(reply, content=single, original_wire_content=wire,
                         wire_normalization=dict(kind='exact_duplicate_collapsed', copies=2, changed_text=False,
                                                 source_sha256=hashlib.sha256(wire.encode()).hexdigest()))
        single = first_turn(reply.get('content') or '')
        if single is not None:
            wire = reply.get('original_wire_content', reply['content'])
            reply = dict(reply, content=single, original_wire_content=wire,
                         wire_normalization=dict(kind='runaway_turns_dropped', changed_text=False,
                                                 dropped_chars=len(reply['content']) - len(single),
                                                 source_sha256=hashlib.sha256(wire.encode()).hexdigest()))
        return dict(reply, transport_retries=failed) if failed else reply


def turn_errors(parsed: dict, n: int, crops_before: int, available: List[str]) -> List[str]:
    return proto.check_step(parsed, n, crops_before, available, BUDGET) + batch.check_turn(parsed['text'])


def disturbed(parsed: dict, views: List[dict], size: tuple, seed: list, n: int, crops_before: int,
              available: List[str], kinds: dict) -> Optional[dict]:
    """A legal turn equal to the generator's own, except for the executed region."""
    source = next(v for v in views if v['view_id'] == parsed['source'])
    action = plan.perturb(parsed['bbox'], source['effective_box_px'], size[0], size[1], seed, kinds)
    if action is None:
        return None
    forced = proto.parse_step(plan.forced_text(parsed['text'], action['bbox'], parsed['source']))
    if turn_errors(forced, n, crops_before, available):
        return None
    return dict(parsed=forced, action=action)


def cap_view(view: dict, side: Optional[int], dest: Path) -> dict:
    """Deliver a view with its longer side reduced to ``side`` (student-scale delivery)."""
    if not side or max(view['shown_px']) <= side:
        return view
    with Image.open(view['path']) as im:
        scale = side / max(im.size)
        small = im.convert('RGB').resize((max(28, round(im.width * scale)), max(28, round(im.height * scale))), Image.Resampling.BICUBIC)
    small.save(dest, 'JPEG', quality=95)
    return dict(view, shown_px=list(small.size), sha256=old.sha(dest), path=str(dest), delivery_max_side=side)


def rollout(parent: Path, p: dict, api, todo: dict, condition: str) -> dict:
    """Autonomous episode; some of the generator's own crop requests are executed off target."""
    aid, out = p['audit_id'], parent/api.role
    odir = out/api.role/'obs'/aid
    odir.mkdir(parents=True, exist_ok=False)
    side = todo.get('view_max_side')
    ro = dict(protocol=proto.PROTOCOL_VERSION, budget=BUDGET,
              views=[cap_view(copy.deepcopy(p['initial_view']), side, odir/'obs0.jpg')],
              steps=[], termination='not_finished', answer=None, export=None)
    with Image.open(p['image']) as im:
        native = im.convert('RGB')
    ordinal = used = 0
    for n in range(1, MAX_STEPS + 1):
        write_json(out/api.role/'checkpoints'/(aid+'.json'), dict(rollout=ro, next_step=n))
        available = [v['view_id'] for v in ro['views']]
        crops_before = len(available) - 1
        reply = api.call(f'{aid}/s{n}', with_condition(old.as_messages(ro, api, p['question']), condition),
                         replicate_id=f"{plan.VERSION}#{todo['variant']}#{api.role}#{aid}#s{n}", temperature=.7, top_p=.9)
        st = dict(step=n, request_sha256=reply['request_sha256'], finish_reason=reply['finish_reason'],
                  raw_text=reply.get('content') or '', crops_before=crops_before, available_before=available,
                  origin='model_continuation', usage=reply.get('usage'),
                  wire_normalization=(reply.get('wire_normalization') or {}).get('kind'))
        ro['steps'].append(st)
        if reply['finish_reason'] != 'stop':
            ro['termination'] = 'api_error' if reply['finish_reason'] == 'error' else 'provider_finish_error'
            break
        parsed = proto.parse_step(st['raw_text'])
        st.update(parsed=parsed, violations=turn_errors(parsed, n, crops_before, available), flags=proto.soft_flags(parsed))
        if st['violations']:
            ro['termination'] = 'protocol_violation'
            break
        if parsed['kind'] == 'answer':
            ro.update(termination='abstain' if parsed['answer'].upper() == 'UNCLEAR' else 'answer', answer=parsed['answer'])
            break
        ordinal += 1
        forced_run = 0
        for earlier in reversed(ro['steps'][:-1]):
            if earlier['origin'] != 'editor_forced':
                break
            forced_run += 1
        change = None
        if plan.due(todo['slots'], ordinal, used, forced_run, crops_before, BUDGET,
                    plan.unit(aid, 'chain', ordinal), todo['chain']):
            change = disturbed(parsed, ro['views'], native.size, [aid, ordinal], n, crops_before, available, todo['kinds'])
        if change:
            used += 0 if forced_run else 1
            st.update(origin='editor_forced', parsed=change['parsed'], flags=proto.soft_flags(change['parsed']),
                      intended=dict(text=parsed['text'], bbox=parsed['bbox'], source=parsed['source']),
                      intervention=dict(change['action'], grounding_ordinal=ordinal))
            parsed = change['parsed']
        view, meta = old.execute_crop(native, ro['views'], parsed, n, odir)
        view = cap_view(view, side, Path(view['path']))
        st.update(meta)
        if any(v['effective_box_px'] == view['effective_box_px'] for v in ro['views']):
            st['flags'].append('repeat_crop')
        ro['views'].append(view)
    if ro['termination'] == 'not_finished':
        ro['termination'] = 'budget_exhausted'
    ro['crops'] = len(ro['views']) - 1
    return ro


def generate(parent: Path, p: dict, api, private: dict) -> dict:
    aid, out = p['audit_id'], parent/api.role
    variant = batch.read(out/'config.json')['v3_variant']
    todo = plan.make_plan(aid, variant)
    condition = plan.condition(variant, todo['arm'])
    ro = rollout(parent, p, api, todo, condition)
    if ro['termination'] in ('answer', 'abstain'):
        # Student view keeps the plain protocol; the construction condition is teacher scaffolding.
        ro['export'] = dict(old.student_export(ro, p['question']), generation_condition=todo['arm'],
                            teacher_condition_removed_from_student_view=True)
    forced = [s for s in ro['steps'] if s['origin'] == 'editor_forced']
    violations = [x for s in ro['steps'] for x in s.get('violations', [])]
    violations += [f for s in ro['steps'] for f in s.get('flags', []) if f in ('coordinate_talk', 'coordinate_unresolved')]
    passed = ro['termination'] == 'answer' and not violations
    gt = batch.score(private['answer_spec'], ro.get('answer') or '')
    mask_ok = bool(ro['export']) and ro['export']['assistant_supervision'] == [s['origin'] != 'editor_forced' for s in ro['steps']]
    eligible = bool(passed and gt['score'] == 1 and mask_ok)
    unscored = passed and gt['score'] is None
    if eligible:
        reason = None
    elif unscored:
        reason = 'unscored_answer_kept_for_review'
    else:
        reason = 'GT_or_flags_conflict' if passed else 'objective_failure:' + ro['termination']
    last_forced = max((s['step'] for s in forced), default=0)
    result = dict(
        audit_id=aid, item_id=p['item_id'], model=api.model, assigned_role=api.role, effort='low',
        original_model=p['original_model'], original_crops=p['original_crops'], rollout=ro, bootstrap={},
        prefix_origin='fresh_' + api.model, interpolation_inserted=bool(forced),
        intervention_not_applicable_reason=None if forced or not todo['slots'] else 'planned_grounding_decision_not_reached_or_no_legal_geometry',
        prompt_condition=todo['arm'], prompt_condition_sha256=hashlib.sha256(condition.encode()).hexdigest(),
        construction=dict(todo, interventions=len(forced), runaway_turns_dropped=sum(s.get('wire_normalization') == 'runaway_turns_dropped' for s in ro['steps']), intervention_steps=[s['step'] for s in forced],
                          kinds=[s['intervention']['kind'] for s in forced],
                          model_crops_after_last_intervention=sum(s['origin'] == 'model_continuation' and bool(s.get('observation'))
                                                                 and s['step'] > last_forced for s in ro['steps']) if forced else None),
        paired_split=private['split'], semantic_review='NOT_DONE', process_quality_unverified=True,
        cpu=dict(passed=passed, reasons=violations),
        objective_checks=dict(protocol_passed=passed, GT_score=gt, editor_supervision_flags_correct=mask_ok,
                              eligible_without_process_review=eligible, actual_token_mask_verified=False,
                              target_4B_loader_verified=False),
        # Unscored answers cannot pass on any model; sending them down the funnel would only spend calls.
        needs_fallback=not eligible and not unscored, fallback_reason=reason,
        config_sha256=old.sha(out/'config.json'),
        provenance='fresh autonomous episode; selected own crop requests executed off target (loss0 turns); no editor text')
    write_json(out/api.role/'results'/(aid+'.json'), result)
    return result


def install_run() -> None:
    base.set_route()
    batch.Transport = V3Transport
    batch.generate = generate


def select(source: Path, pilot: bool) -> Dict[str, str]:
    """Which tasks get a v3 episode. Reviewed recoveries and ordinary short solves are kept as they are.

    A pilot compares construction variants on the same train-split tasks and ignores those dispositions.
    """
    if pilot:
        return {p['audit_id']: 'construct_v3' if p['split'] == 'train' else 'pilot_train_only'
                for p in batch.read(source/'SELECTION_PRIVATE.json')}
    queues = batch.read(QUEUES)
    disposition = {aid: name for name, ids in queues.items() if not name.startswith('_') for aid in ids}
    ids = [s['audit_id'] for s in batch.read(source/'SCOPE_ALL_4040.json')]
    if len(ids) != 4040 or set(ids) != set(disposition):
        raise ValueError('Audit queues do not cover the 4040 scope')
    return {aid: 'keep_existing:' + disposition[aid] if disposition[aid] in KEEP_EXISTING else 'construct_v3' for aid in ids}


def build(parent: Path, source: Path, limit: Optional[int], exclude: Iterable[Path], variant: dict, pilot: bool) -> dict:
    """Create a v3 campaign directory. Sends nothing."""
    if parent.exists():
        raise ValueError('Fresh v3 directory required')
    batch.verify(source)
    decisions = select(source, pilot)
    for other in exclude:
        for row in batch.read(other/'SCOPE_ALL_4040.json'):
            decisions[row['audit_id']] = 'issued_in:' + other.name
    ids = sorted((a for a, d in decisions.items() if d == 'construct_v3'), key=lambda a: plan.unit(a, 'order'))
    if limit:
        for aid in ids[limit:]:
            decisions[aid] = 'construct_v3_later_wave'
        ids = ids[:limit]
    chosen = set(ids)
    parent.mkdir(parents=True)
    scope_all = batch.read(source/'SCOPE_ALL_4040.json')
    write_json(parent/'SCOPE_ALL_4040.json', [s for s in scope_all if s['audit_id'] in chosen])
    write_json(parent/'SELECTION_PRIVATE.json', [p for p in batch.read(source/'SELECTION_PRIVATE.json') if p['audit_id'] in chosen])
    write_json(parent/'SOURCE_PIXEL_HASHES.json', batch.read(source/'SOURCE_PIXEL_HASHES.json'))
    original = batch.read(source/'INPUT_HASHES.json')
    hashes = {}
    (parent/'inputs').mkdir()
    for aid in ids:
        src = source/'inputs'/(aid+'.json')
        if old.sha(src) != original[str(src)]:
            raise ValueError('Input changed')
        dst = parent/'inputs'/(aid+'.json')
        shutil.copyfile(src, dst)
        hashes[str(dst)] = old.sha(dst)
    write_json(parent/'INPUT_HASHES.json', hashes)
    plans = {aid: plan.make_plan(aid, variant) for aid in ids}
    write_json(parent/'V3_SELECTION.json', dict(at=batch.now(), source=str(source), decisions=decisions,
                                                counts=dict(collections.Counter(d.split(':')[0] for d in decisions.values()))))
    write_json(parent/'V3_PLAN.json', plans)
    shutil.copyfile(source/base.PROXY_RECORD, parent/base.PROXY_RECORD)
    cfg = batch.read(source/'config.json')
    primary = variant.get('primary', 'sol')
    cfg.update(version=plan.VERSION, v3_variant=variant, pilot=pilot, primary_role=primary, tasks=len(ids), authorization=AUTHORIZATION, built_from=str(source), created_at=batch.now(),
               deadline=time.time() + DEADLINE_DAYS * 86400, budget_crops=BUDGET, max_HTTP_per_task_per_role=MAX_STEPS,
               semantic_verified=False, process_review='NOT_DONE',
               retry_policy='one episode per model/task in this version; no same-model re-roll of valid, format or technical outcomes',
               fallback='objective failure with a scorable answer -> next model (Sol -> K3 -> Gemini); no GT or error details passed on',
               direct_policy='answering without a crop is allowed in both arms and kept as a short path',
               tool_prefix_policy='none: every episode starts fresh from the question and original image',
               intervention='own crop request executed off target (shift/clip/tight) at hash-planned grounding ordinals; '
                            'caption kept verbatim; never right after a disturbed crop; no editor text, no GT, no future view',
               masks='disturbed turns loss0 (body and EOS); all other turns supervised; span masks not used',
               construction_prompt=dict(conditions={a: plan.condition(variant, a) for a in variant['arms']},
                                        student_view='plain caption-v5.1 system; condition removed'),
               plan_weights=dict(coverage_limits=plan.COVERAGE_LIMITS),
               wire_normalization=dict(kinds=['exact_duplicate_collapsed', 'runaway_turns_dropped'], original_kept_in_receipt=True))
    cfg['frozen_files'] = dict(cfg['frozen_files'])
    for path in (*WRAPPERS, QUEUES, source/'config.json', source/'CONFIG.sha256', parent/'SCOPE_ALL_4040.json',
                 parent/'SELECTION_PRIVATE.json', parent/'SOURCE_PIXEL_HASHES.json', parent/'INPUT_HASHES.json',
                 parent/'V3_SELECTION.json', parent/'V3_PLAN.json'):
        cfg['frozen_files'][str(path)] = old.sha(path)
    write_json(parent/'config.json', cfg)
    (parent/'CONFIG.sha256').write_text(old.sha(parent/'config.json')+'\n')
    held = 'funnel opens after the primary wave of this version has settled and been inspected'
    write_json(parent/campaign.REGISTRY, dict(
        version=plan.VERSION, campaign_parent=str(parent), authorized_by=AUTHORIZATION,
        not_before_utc=dict(sol=batch.now(), k3=batch.now(), gemini=cfg['gemini_not_before_utc']),
        blocked={r: held if primary == 'sol' else 'pilot with primary ' + primary for r in batch.ROLES if r != primary},
        waves=[], acknowledged_stopped_waves=[]))
    batch.verify(parent)
    summary = collections.Counter
    return dict(parent=str(parent), tasks=len(ids), decisions=dict(summary(d.split(':')[0] for d in decisions.values())),
                arms=dict(summary(p['arm'] for p in plans.values())),
                planned_interventions=dict(sorted(summary(p['planned_interventions'] for p in plans.values()).items())),
                generation_calls=0)


def prepare_primary_config(parent: Path, loc: Path) -> set:
    """Point a never-launched Sol wave at the owned retry proxy and freeze the wrapper chain."""
    if any((loc/'sol'/n).exists() for n in ('LAUNCH.json', 'RUN_STATE.json', 'PAUSE')):
        raise ValueError('Never edit an issued wave')
    issued, _ = campaign.inventory(parent)
    proxy = batch.read(parent/base.PROXY_RECORD)
    upcfg = Path(proxy['runtime_dir'])/'cliproxy.yaml'
    if old.sha(upcfg) != proxy['config_sha256']:
        raise ValueError('Owned proxy changed')
    cfg = batch.read(loc/'config.json')
    cfg['models']['sol'].update(model=SOL, endpoint=base.ENDPOINT, workers=base.WORKERS, rpm=base.RPM, effort='low',
                                allowed_returned_models=[SOL])
    cfg['actual_primary_model'] = SOL
    cfg['provider_route'] = 'native Pro-only Codex GPT5.6Sol via retry-enabled owned proxy; no Cursor/xAI'
    cfg['proxy_retry'] = dict(request_retry=3, max_retry_interval_seconds=30, transient_error_cooldown='disabled')
    cfg['wire_normalization'] = dict(kind='exact_duplicate_collapsed', min_turn_chars=dedup.MIN_TURN_CHARS)
    cfg['client_transport_retry'] = dict(code=backoff.RETRY_CODE, backoff_seconds=list(backoff.BACKOFF_SECONDS),
                                         only_when_no_model_output=True, counted_in_ledger=False)
    cfg['frozen_files'][str(upcfg)] = old.sha(upcfg)
    write_json(loc/'config.json', cfg)
    (loc/'CONFIG.sha256').write_text(old.sha(loc/'config.json')+'\n')
    branch = dict(cfg, frozen_files=dict(cfg['frozen_files'], **{str(loc/'config.json'): old.sha(loc/'config.json')}))
    write_json(loc/'sol/config.json', branch)
    (loc/'sol/CONFIG.sha256').write_text(old.sha(loc/'sol/config.json')+'\n')
    batch.verify(loc, 'sol')
    ids = {row['audit_id'] for row in batch.read(loc/'SCOPE_ALL_4040.json')}
    if ids & issued['sol']:
        raise ValueError('Task already issued in this version')
    return ids


def launch_wave(parent: Path, loc: Path, role: str) -> dict:
    """Funnel wave (K3 or Gemini) run by this constructor instead of the 2026-10-11 one."""
    cfg = batch.verify(loc, role)
    out = loc/role
    if time.time() > cfg['deadline'] or (parent/'PAUSE').exists() or (loc/'PAUSE').exists():
        raise ValueError('Deadline elapsed or campaign paused')
    if any((out/n).exists() for n in ('LAUNCH.json', 'RUN_STATE.json', 'PAUSE')):
        raise ValueError('Never relaunch an issued wave')
    if batch.read(parent/campaign.REGISTRY).get('blocked', {}).get(role):
        raise ValueError('Role blocked')
    if role == 'gemini':
        batch.start_gemini_proxy(loc)
    command = ['${PYTHON}', '-u', str(THIS_FILE), 'run', '--parent', str(parent),
               '--wave', str(loc), '--role', role]
    with (out/'worker.log').open('ab') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out/'LAUNCH.json', dict(pid=proc.pid, command=command, at=batch.now(), first_real_receipt_pending=True))
    return dict(launched=True, role=role, tasks=cfg['tasks'], first_real_receipt_pending=True)


_funnel_ids = campaign.eligible_ids


def eligible_ids(parent: Path, role: str) -> List[str]:
    """Funnel rule, except that a pilot may name a non-Sol model as the first generator."""
    if role != 'sol' and batch.read(parent/'config.json').get('primary_role') == role:
        issued, _ = campaign.inventory(parent)
        return [s['audit_id'] for s in batch.read(parent/'SCOPE_ALL_4040.json') if s['audit_id'] not in issued[role]]
    return _funnel_ids(parent, role)


base.prepare_primary_config = prepare_primary_config
campaign.launch_wave = launch_wave
campaign.eligible_ids = eligible_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'tick', 'run'])
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--wave', type=Path)
    parser.add_argument('--role', choices=batch.ROLES, default='sol')
    parser.add_argument('--limit', type=int, help='build only the first N tasks of the hash order (calibration)')
    parser.add_argument('--exclude', type=Path, action='append', default=[], help='v3 campaign whose tasks are already issued')
    parser.add_argument('--variant-file', type=Path, help='JSON object of fields overriding the default variant')
    parser.add_argument('--pilot', action='store_true', help='variant comparison on train tasks; ignores keep dispositions')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    target = args.parent.resolve()
    if args.action == 'build':
        print(json.dumps(build(target, args.source.resolve(), args.limit, [e.resolve() for e in args.exclude],
                               dict(plan.DEFAULT_VARIANT, **(batch.read(args.variant_file) if args.variant_file else {})), args.pilot)))
    elif args.action == 'tick':
        print(json.dumps(base.tick(target, args.launch)))
    else:
        install_run()
        campaign.run_wave(target, args.wave.resolve(), args.role)
