"""Construction v4 campaign: skill-card protocol (caption v6), forced interpolation, GPT-6 Astra as generator.

User decision 2026-10-13: regenerate all 4040 trajectories with the strongest model (GPT-6 Astra), with a
redesigned caption, a written specification the model knows in advance, stronger forced interpolation and
longer trajectories; no K3 or Gemini. Planning lives in ``interp_v4_plan_20261013``, the student protocol
in ``caption_v6``.

This is a new construction version in its own campaign directory with its own ledger. Earlier campaigns,
their results, GT, splits and frozen sources are read only. One episode per model and task. The transport,
view delivery and launch plumbing are the ones validated in v3r2 and are imported, not copied.
"""
import argparse
import collections
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image

import caption_v6 as proto
import interp_v3r2_campaign_20261012 as r2
import interp_v4_plan_20261013 as plan
from tiered_transport import ROOT, write_json

batch, base, campaign, old, backoff, dedup = r2.batch, r2.base, r2.campaign, r2.old, r2.backoff, r2.dedup
THIS_FILE = Path(__file__).resolve()
PLAN_FILE = Path(plan.__file__).resolve()
PROTO_FILE = Path(proto.__file__).resolve()
SOURCE = campaign.PARENT
PARENT = ROOT/'obs/runs/interp_v4_astra_20261013'
MODEL = 'gpt-6-astra'
DEADLINE_DAYS = 5
WRAPPERS = (THIS_FILE, PLAN_FILE, PROTO_FILE, *r2.WRAPPERS)
AUTHORIZATION = ('2026-10-13 user: regenerate all trajectories with GPT-6 Astra; redesign the caption and write down '
                 'the specification the model must know in advance; force the interpolation; make trajectories '
                 'longer; quota is sufficient. No K3 or Gemini. New construction version, own ledger.')


def text_msg(role: str, text: str) -> dict:
    return dict(role=role, content=[dict(type='text', text=text)])


def as_messages(ro: dict, api, question: str) -> List[dict]:
    """Only the question, the views actually delivered, and the executed assistant turns."""
    views, budget = ro['views'], ro['budget']
    head, tail = proto.first_user_text(question, views[0]['source_px'], views[0]['shown_px'], budget)
    msgs = [text_msg('system', proto.to_api_tags(proto.SYSTEM)),
            dict(role='user', content=[dict(type='text', text=head), api.image(views[0]['path']), dict(type='text', text=tail)])]
    available = ['original_image']
    for s in ro['steps']:
        msgs.append(text_msg('assistant', proto.to_api_tags(s['parsed']['text'])))
        if s.get('observation'):
            v = next(v for v in views if v['view_id'] == s['observation'])
            available.append(v['view_id'])
            head, tail = proto.receipt_text(s['step'], len(available) - 1, v['parent'], v['fov_original'], v['source_px'],
                                            v['shown_px'], available, len(available) - 1, budget)
            msgs.append(dict(role='user', content=[dict(type='text', text=head), api.image(v['path']), dict(type='text', text=tail)]))
    return msgs


def student_export(ro: dict, question: str) -> dict:
    turns, available, budget = [], ['original_image'], ro['budget']
    for s in ro['steps']:
        turns.append(dict(role='gpt', text=s['parsed']['text']))
        if s.get('observation'):
            v = next(v for v in ro['views'] if v['view_id'] == s['observation'])
            available.append(v['view_id'])
            head, tail = proto.receipt_text(s['step'], len(available) - 1, v['parent'], v['fov_original'], v['source_px'],
                                            v['shown_px'], available, len(available) - 1, budget)
            turns.append(dict(role='human', head=head, tail=tail))
    export = proto.to_sharegpt(question, turns, [v['path'] for v in ro['views']], ro['views'][0]['source_px'],
                               ro['views'][0]['shown_px'], budget)
    export['assistant_supervision'] = [s['origin'] != 'editor_forced' for s in ro['steps']]
    return export


def turn_errors(parsed: dict, n: int, crops_before: int, available: List[str], budget: int) -> List[str]:
    return proto.check_step(parsed, n, crops_before, available, budget) + batch.check_turn(parsed['text'])


def disturbed(parsed: dict, views: List[dict], size: tuple, seed: list, n: int, crops_before: int,
              available: List[str], kinds: dict, budget: int) -> Optional[dict]:
    """A legal turn equal to the generator's own, except for the executed region."""
    source = next(v for v in views if v['view_id'] == parsed['source'])
    action = plan.perturb(parsed['bbox'], source['effective_box_px'], size[0], size[1], seed, kinds)
    if action is None:
        return None
    forced = proto.parse_step(plan.forced_text(parsed['text'], action['bbox'], parsed['source']))
    if turn_errors(forced, n, crops_before, available, budget):
        return None
    return dict(parsed=forced, action=action)


def rollout(parent: Path, p: dict, api, todo: dict, condition: str) -> dict:
    """Autonomous episode; some of the generator's own crop requests are executed off target."""
    aid, out = p['audit_id'], parent/api.role
    odir = out/api.role/'obs'/aid
    odir.mkdir(parents=True, exist_ok=False)
    side, budget = todo['view_max_side'], todo['budget']
    ro = dict(protocol=proto.PROTOCOL_VERSION, budget=budget,
              views=[r2.cap_view(copy.deepcopy(p['initial_view']), side, odir/'obs0.jpg')],
              steps=[], termination='not_finished', answer=None, export=None)
    with Image.open(p['image']) as im:
        native = im.convert('RGB')
    ordinal = used = 0
    for n in range(1, budget + 2):
        write_json(out/api.role/'checkpoints'/(aid+'.json'), dict(rollout=ro, next_step=n))
        available = [v['view_id'] for v in ro['views']]
        crops_before = len(available) - 1
        reply = api.call(f'{aid}/s{n}', r2.with_condition(as_messages(ro, api, p['question']), condition),
                         replicate_id=f"{plan.VERSION}#{todo['variant']}#{api.model}#{aid}#s{n}", temperature=.7, top_p=.9)
        st = dict(step=n, request_sha256=reply['request_sha256'], finish_reason=reply['finish_reason'],
                  raw_text=reply.get('content') or '', crops_before=crops_before, available_before=available,
                  origin='model_continuation', usage=reply.get('usage'),
                  wire_normalization=(reply.get('wire_normalization') or {}).get('kind'))
        ro['steps'].append(st)
        if reply['finish_reason'] != 'stop':
            ro['termination'] = 'api_error' if reply['finish_reason'] == 'error' else 'provider_finish_error'
            break
        parsed = proto.parse_step(st['raw_text'])
        st.update(parsed=parsed, violations=turn_errors(parsed, n, crops_before, available, budget), flags=proto.soft_flags(parsed))
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
        if plan.due(todo['slots'], ordinal, used, forced_run, crops_before, budget,
                    plan.unit(aid, 'chain', ordinal), todo['chain']):
            change = disturbed(parsed, ro['views'], native.size, [aid, ordinal], n, crops_before, available, todo['kinds'], budget)
        if change:
            used += 0 if forced_run else 1
            st.update(origin='editor_forced', parsed=change['parsed'], flags=proto.soft_flags(change['parsed']),
                      intended=dict(text=parsed['text'], bbox=parsed['bbox'], source=parsed['source']),
                      intervention=dict(change['action'], grounding_ordinal=ordinal))
            parsed = change['parsed']
        view, meta = old.execute_crop(native, ro['views'], parsed, n, odir)
        view = r2.cap_view(view, side, Path(view['path']))
        st.update(meta)
        if any(v['effective_box_px'] == view['effective_box_px'] for v in ro['views']):
            st['flags'].append('repeat_crop')
        ro['views'].append(view)
    if ro['termination'] == 'not_finished':
        ro['termination'] = 'budget_exhausted'
    ro['crops'] = len(ro['views']) - 1
    return ro


def reactions(steps: List[dict]) -> List[dict]:
    """For every disturbed crop: the verdict and skill of the turn that received it (structural, no pixels)."""
    out = []
    for i, s in enumerate(steps[:-1]):
        if s['origin'] == 'editor_forced':
            nxt = steps[i + 1].get('intended') or {}
            seen = proto.parse_step(nxt['text']) if nxt else steps[i + 1].get('parsed') or {}
            out.append(dict(step=s['step'], kind=s['intervention']['kind'], elicits=s['intervention']['elicits'],
                            coverage=s['intervention']['intended_region_coverage'],
                            next_verdict=seen.get('verdict'), next_skill=seen.get('behavior')))
    return out


def generate(parent: Path, p: dict, api, private: dict) -> dict:
    aid, out = p['audit_id'], parent/api.role
    variant = batch.read(out/'config.json')['v4_variant']
    todo = plan.make_plan(aid, variant)
    condition = plan.condition(variant, todo['arm'])
    ro = rollout(parent, p, api, todo, condition)
    if ro['termination'] in ('answer', 'abstain'):
        ro['export'] = dict(student_export(ro, p['question']), generation_condition=todo['arm'],
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
    parsed = [s['parsed'] for s in ro['steps'] if s.get('parsed')]
    result = dict(
        audit_id=aid, item_id=p['item_id'], model=api.model, assigned_role=api.role, effort='low',
        original_model=p['original_model'], original_crops=p['original_crops'], rollout=ro, bootstrap={},
        prefix_origin='fresh_' + api.model, interpolation_inserted=bool(forced),
        intervention_not_applicable_reason=None if forced or not todo['slots'] else 'planned_crop_request_not_reached_or_no_legal_geometry',
        prompt_condition=todo['arm'], prompt_condition_sha256=hashlib.sha256(condition.encode()).hexdigest(),
        construction=dict(todo, interventions=len(forced),
                          runaway_turns_dropped=sum(s.get('wire_normalization') == 'runaway_turns_dropped' for s in ro['steps']),
                          intervention_steps=[s['step'] for s in forced], kinds=[s['intervention']['kind'] for s in forced],
                          reactions=reactions(ro['steps']),
                          skills=[q.get('behavior') for q in parsed], verdicts=[q.get('verdict') for q in parsed],
                          model_crops_after_last_intervention=sum(s['origin'] == 'model_continuation' and bool(s.get('observation'))
                                                                 and s['step'] > last_forced for s in ro['steps']) if forced else None),
        paired_split=private['split'], semantic_review='NOT_DONE', process_quality_unverified=True,
        cpu=dict(passed=passed, reasons=violations),
        objective_checks=dict(protocol_passed=passed, GT_score=gt, editor_supervision_flags_correct=mask_ok,
                              eligible_without_process_review=eligible, actual_token_mask_verified=False,
                              target_4B_loader_verified=False),
        needs_fallback=not eligible and not unscored, fallback_reason=reason,
        config_sha256=old.sha(out/'config.json'),
        provenance='fresh autonomous episode under caption v6; selected own crop requests executed off target (loss0 turns); no editor text')
    write_json(out/api.role/'results'/(aid+'.json'), result)
    return result


def build(parent: Path, source: Path, limit: Optional[int], exclude: Iterable[Path], variant: dict) -> dict:
    """Create a v4 campaign directory over all 4040 tasks (minus those issued elsewhere). Sends nothing."""
    if parent.exists():
        raise ValueError('Fresh v4 directory required')
    batch.verify(source)
    decisions = {s['audit_id']: 'construct_v4' for s in batch.read(source/'SCOPE_ALL_4040.json')}
    if len(decisions) != 4040:
        raise ValueError('Source scope is not the 4040 tasks')
    for other in exclude:
        for row in batch.read(other/'SCOPE_ALL_4040.json'):
            decisions[row['audit_id']] = 'issued_in:' + other.name
    ids = sorted((a for a, d in decisions.items() if d == 'construct_v4'), key=lambda a: plan.unit(a, 'order'))
    if limit:
        for aid in ids[limit:]:
            decisions[aid] = 'construct_v4_later_wave'
        ids = ids[:limit]
    chosen = set(ids)
    parent.mkdir(parents=True)
    write_json(parent/'SCOPE_ALL_4040.json', [s for s in batch.read(source/'SCOPE_ALL_4040.json') if s['audit_id'] in chosen])
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
    write_json(parent/'V4_SELECTION.json', dict(at=batch.now(), source=str(source), decisions=decisions,
                                                counts=dict(collections.Counter(d.split(':')[0] for d in decisions.values()))))
    write_json(parent/'V4_PLAN.json', plans)
    shutil.copyfile(source/base.PROXY_RECORD, parent/base.PROXY_RECORD)
    cfg = batch.read(source/'config.json')
    cfg.update(version=plan.VERSION, v4_variant=variant, primary_role='sol', primary_model=MODEL, tasks=len(ids),
               authorization=AUTHORIZATION, built_from=str(source), created_at=batch.now(),
               deadline=time.time() + DEADLINE_DAYS * 86400, budget_crops='per task: ' + json.dumps(variant['budget']),
               max_HTTP_per_task_per_role=proto.MAX_CROPS + 1, student_protocol=proto.PROTOCOL_VERSION,
               student_system_sha256=hashlib.sha256(proto.SYSTEM.encode()).hexdigest(),
               semantic_verified=False, process_review='NOT_DONE',
               role_note='the campaign slot is named "sol"; the model in it is ' + MODEL,
               retry_policy='one episode per model/task in this version; no same-model re-roll of valid, format or technical outcomes',
               fallback='none inside this campaign; K3 and Gemini are not used (user decision 2026-10-13)',
               direct_policy='answering at Step 1 is allowed in both arms and kept as a short path',
               tool_prefix_policy='none: every episode starts fresh from the question and original image',
               intervention='own crop request executed off target (shift/clip/tight/wide/far) at hash-planned request ordinals; '
                            'caption kept verbatim; at most two in a row; no editor text, no GT, no future view',
               masks='disturbed turns loss0 (body and EOS); all other turns supervised; span masks not used',
               construction_prompt=dict(conditions={a: plan.condition(variant, a) for a in variant['arms']},
                                        student_view='caption v6 system only; construction condition removed'),
               plan_weights=dict(coverage_limits=plan.COVERAGE_LIMITS, elicits=plan.ELICITS),
               wire_normalization=dict(kinds=['exact_duplicate_collapsed', 'runaway_turns_dropped'], original_kept_in_receipt=True))
    cfg['frozen_files'] = dict(cfg['frozen_files'])
    for path in (*WRAPPERS, source/'config.json', source/'CONFIG.sha256', parent/'SCOPE_ALL_4040.json',
                 parent/'SELECTION_PRIVATE.json', parent/'SOURCE_PIXEL_HASHES.json', parent/'INPUT_HASHES.json',
                 parent/'V4_SELECTION.json', parent/'V4_PLAN.json'):
        cfg['frozen_files'][str(path)] = old.sha(path)
    write_json(parent/'config.json', cfg)
    (parent/'CONFIG.sha256').write_text(old.sha(parent/'config.json')+'\n')
    stopped = 'K3 and Gemini are not used (user decision 2026-10-13)'
    write_json(parent/campaign.REGISTRY, dict(
        version=plan.VERSION, campaign_parent=str(parent), authorized_by=AUTHORIZATION,
        not_before_utc=dict(sol=batch.now(), k3=batch.now(), gemini=cfg['gemini_not_before_utc']),
        blocked=dict(k3=stopped, gemini=stopped), waves=[], acknowledged_stopped_waves=[]))
    batch.verify(parent)
    count = collections.Counter
    return dict(parent=str(parent), tasks=len(ids), decisions=dict(count(d.split(':')[0] for d in decisions.values())),
                arms=dict(count(p['arm'] for p in plans.values())),
                planned_interventions=dict(sorted(count(p['planned_interventions'] for p in plans.values()).items())),
                budgets=dict(sorted(count(p['budget'] for p in plans.values()).items())),
                view_max_side=dict(sorted(count(p['view_max_side'] for p in plans.values()).items())), generation_calls=0)


def prepare_primary_config(parent: Path, loc: Path) -> set:
    """Point a never-launched wave at the owned retry proxy with the v4 model and freeze the wrapper chain."""
    if any((loc/'sol'/n).exists() for n in ('LAUNCH.json', 'RUN_STATE.json', 'PAUSE')):
        raise ValueError('Never edit an issued wave')
    issued, _ = campaign.inventory(parent)
    proxy = batch.read(parent/base.PROXY_RECORD)
    upcfg = Path(proxy['runtime_dir'])/'cliproxy.yaml'
    if old.sha(upcfg) != proxy['config_sha256']:
        raise ValueError('Owned proxy changed')
    cfg = batch.read(loc/'config.json')
    cfg['models']['sol'].update(model=base.MODEL, endpoint=base.ENDPOINT, workers=base.WORKERS, rpm=base.RPM, effort='low',
                                call_cap=(proto.MAX_CROPS + 2) * cfg['tasks'], allowed_returned_models=[base.MODEL])
    cfg['actual_primary_model'] = base.MODEL
    cfg['provider_route'] = base.MODEL + ' via the retry-enabled owned ChatGPT-OAuth proxy; no Cursor/xAI, no K3, no Gemini'
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


def install(model: str = MODEL) -> None:
    """Route the campaign slot to ``model`` and make launched workers start from the calling file."""
    base.MODEL = model
    base.ALLOWED_RETURNED_MODELS = [model]
    base.AUTHORIZATION = AUTHORIZATION
    base.prepare_primary_config = prepare_primary_config
    base.__file__ = str(THIS_FILE)


def install_run() -> None:
    base.set_route()
    batch.Transport = r2.V3Transport
    batch.generate = generate


install()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'tick', 'run'])
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--wave', type=Path)
    parser.add_argument('--limit', type=int, help='build only the first N tasks of the hash order (calibration)')
    parser.add_argument('--exclude', type=Path, action='append', default=[], help='v4 campaign whose tasks are already issued')
    parser.add_argument('--variant-file', type=Path, help='JSON object of fields overriding the default variant')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    target = args.parent.resolve()
    if args.action == 'build':
        variant = dict(plan.DEFAULT_VARIANT, **(batch.read(args.variant_file) if args.variant_file else {}))
        print(json.dumps(build(target, SOURCE, args.limit, [e.resolve() for e in args.exclude], variant)))
    elif args.action == 'tick':
        print(json.dumps(base.tick(target, args.launch)))
    else:
        install_run()
        campaign.run_wave(target, args.wave.resolve(), 'sol')
