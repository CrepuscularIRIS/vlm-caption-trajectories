"""Independent cc review without a two-correct-generator prerequisite.

Structure extraction and each prefix-limited blind fact use different CLI processes.
The frozen final answer read is also independent. Automatic PASS still requires manual KEEP.
"""
import json
import re
from pathlib import Path

from audit_v51 import (BLIND_SYSTEM, REVIEW_SYSTEM, REVIEW_USER, FACT_USER, BLIND_USER,
                       COMPARE_USER, first_json, parse_answer_line, schema_valid, review_passes, coord_flags)
from corpus_scoring import score
from tiered_transport import BudgetStop, write_json


def fact_views(ro, fact):
    n = fact.get('step')
    if type(n) is not int or not 1 <= n <= len(ro['steps']): raise ValueError('invalid_fact_step')
    ids = fact.get('views')
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
        raise ValueError('invalid_fact_views')
    before = set(ro['steps'][n - 1]['available_before'])
    if not set(ids) <= before: raise ValueError('future_or_unknown_fact_view')
    by_id = {v['view_id']: v for v in ro['views']}
    if not set(ids) <= by_id.keys(): raise ValueError('missing_fact_view')
    return [by_id[v] for v in dict.fromkeys(ids)]


def strict_review(p, ro):
    if not schema_valid(p): return ['review_schema_invalid']
    reasons = []
    nsteps = len(ro['steps'])
    valid_step = lambda n: type(n) is int and 1 <= n <= nsteps
    for key in ('instance_switch', 'unsupported_flip'):
        if not all(valid_step(n) for n in p[key]): reasons.append('invalid_' + key)
    for c in p['contradicted']:
        if not isinstance(c, dict) or not valid_step(c.get('step')) or not isinstance(c.get('text'), str) or not isinstance(c.get('view'), str):
            reasons.append('invalid_contradicted')
    observed_updates = {(s['step'], u['kind']) for s in ro['steps'] for u in s.get('parsed', {}).get('updates', [])}
    covered = set()
    for u in p['updates']:
        if not (valid_step(u.get('step')) and u.get('kind') in ('HOLD', 'REVISE') and type(u.get('supported')) is bool
                and (u.get('dependents_withdrawn') is None or type(u.get('dependents_withdrawn')) is bool)):
            reasons.append('invalid_update'); continue
        covered.add((u['step'], u['kind']))
    if covered != observed_updates: reasons.append('missing_or_spurious_update_review')
    expected_fallback = {str(s['step']) for s in ro['steps'] if s.get('parsed', {}).get('behavior') == 'fallback'}
    if set(p['fallback_real']) != expected_fallback or any(type(v) is not bool for v in p['fallback_real'].values()):
        reasons.append('missing_or_invalid_fallback_review')
    for f in p['key_facts']:
        try: fact_views(ro, f)
        except ValueError as exc: reasons.append(str(exc))
        if not f.get('question', '').strip() or len(f.get('claimed', '')) > 160:
            reasons.append('invalid_key_fact_question_or_value')
        if re.match(r'\s*(is|are|does|do|did|can|has|have)\b', f.get('question', ''), re.I):
            reasons.append('leading_yes_no_fact')
    reasons += review_passes(dict(parsed=p))[1]
    return sorted(set(reasons))


def cpu_reasons(result):
    ro = result['rollout']
    reasons = []
    if result['score'].get('score') != 1 or ro['termination'] != 'answer': reasons.append('final_answer_not_correct')
    if any(s.get('violations') for s in ro['steps']): reasons.append('protocol_violation')
    reasons += sorted(coord_flags(ro))
    return reasons


def independent_compare(reviewer, label, claimed, read, question, kind='short', n_options=None):
    if read is None: return 'UNPARSED'
    if 'UNREADABLE' in read.upper(): return 'INSUFFICIENT'
    if score(dict(kind=kind, gold=claimed, n_options=n_options), read)['score'] == 1: return 'SUPPORTED'
    if kind == 'letter': return 'CONTRADICTED'
    # Claim arrives only AFTER the image reader is finished, in a separate text-only process.
    receipt = reviewer.call(label + '/compare', 'Compare a frozen independent reading with a short claim. No images or outside knowledge.',
                            COMPARE_USER.format(question=question, claimed=claimed, read=read), [])
    if not receipt['valid']: return 'INVALID_RECEIPT'
    word = receipt['text'].strip().strip('`.').upper()
    return {'AGREE': 'SUPPORTED', 'DISAGREE': 'CONTRADICTED', 'UNCLEAR': 'INSUFFICIENT'}.get(word, 'UNPARSED')


def audit_candidate(reviewer, result):
    item, ro = result['item'], result['rollout']
    prefix = item['item_id'] + '/' + result['role']
    if result['model'].startswith('claude'): raise ValueError('Generator cannot review itself')
    notes = '\n'.join(f"Step {s['step']} (available before: {', '.join(s['available_before'])}): {s['parsed']['caption']}" for s in ro['steps'])
    prompt = REVIEW_USER.format(question=item['question'], notes=notes)
    prompt += ('\nPrefer one decisive instance-attribute binding fact; use two or three only if genuinely needed. '
               'Report every HOLD/REVISE line and every fallback. Never support an early statement with a later view. '
               'The key-fact question must be open and must not contain its claimed value or offer suggested answers. '
               'Do not excuse guessed detail just because the final answer looks plausible.')
    sr = reviewer.call(prefix + '/structure', REVIEW_SYSTEM, prompt, ro['views'])
    out = dict(structure_receipt=sr['request_sha256'], structure=first_json(sr.get('text')),
               facts=[], reasons=[], model=reviewer.model)
    if not sr['valid']:
        out['reasons'] = ['structure_transport_invalid']; return out
    out['reasons'] = strict_review(out['structure'], ro)
    if out['reasons']: return out
    for k, fact in enumerate(out['structure']['key_facts']):
        views = fact_views(ro, fact)
        label = prefix + f'/fact{k}'
        # Deliberately excludes item, question options, notes, claimed, score and gold.
        br = reviewer.call(label, BLIND_SYSTEM, FACT_USER.format(question=fact['question']), views)
        read = parse_answer_line(br.get('text'))
        status = independent_compare(reviewer, label, fact['claimed'], read, fact['question']) if br['valid'] else 'INVALID_RECEIPT'
        out['facts'].append(dict(fact=fact, read=read, status=status, receipt=br['request_sha256']))
        if status != 'SUPPORTED': out['reasons'].append('key_fact_' + status.lower())
    if out['reasons']: return out
    # Full set cited at commit: no old last-three-view truncation.
    commit = ro['steps'][-1]
    ids = commit['parsed'].get('cited', []) or [ro['views'][-1]['view_id']]
    views = fact_views(ro, dict(step=commit['step'], views=ids))
    kind = item['answer_spec']['kind']
    fmt = 'one option letter' if kind == 'letter' else 'a short answer'
    br = reviewer.call(prefix + '/final_blind', BLIND_SYSTEM, BLIND_USER.format(question=item['question'], fmt=fmt), views)
    read = parse_answer_line(br.get('text'))
    agreement = independent_compare(reviewer, prefix + '/final_blind', ro['answer'], read, item['question'],
                                    kind, item['answer_spec'].get('n_options')) if br['valid'] else 'INVALID_RECEIPT'
    out['final_blind'] = dict(read=read, status=agreement, receipt=br['request_sha256'], views=ids)
    if agreement != 'SUPPORTED': out['reasons'].append('final_blind_' + agreement.lower())
    return out


def audit_item(out, config, reviewer, item):
    dest = out / 'audit' / (item['item_id'] + '.json')
    if dest.exists(): return json.loads(dest.read_text())
    candidates, excluded = [], {}
    for role in config['generators']:
        p = out / role / 'results' / (item['item_id'] + '.json')
        if not p.exists(): continue
        r = json.loads(p.read_text())
        why = cpu_reasons(r)
        if why: excluded[role] = why
        else: candidates.append(r)
    # Preserve Sol priority; no reward for taking needless crops or uttering rare labels.
    order = {'sol': 0, 'k3': 1, 'glm': 2}
    candidates.sort(key=lambda r: (sum(len(s.get('flags', [])) for s in r['rollout']['steps']), order[r['role']]))
    sidecar = dict(item_id=item['item_id'], group_id=item['group_id'], mode=config['selection'],
                   cpu_excluded=excluded, cpu_candidates=[c['role'] for c in candidates],
                   decision='NO_ELIGIBLE_CANDIDATE', manual_required=True, exportable=False)
    if candidates:
        candidate = candidates[0]
        sidecar['selected_role'] = candidate['role']
        sidecar['result_path'] = str(out / candidate['role'] / 'results' / (item['item_id'] + '.json'))
        try:
            sidecar['review'] = audit_candidate(reviewer, candidate)
            sidecar['decision'] = 'AUTO_PASS_PENDING_MANUAL' if not sidecar['review']['reasons'] else 'REVIEW_REJECTED'
        except BudgetStop as exc:
            sidecar['decision'], sidecar['reason'] = 'PENDING_AUDIT_BUDGET', str(exc)
    write_json(dest, sidecar)
    print(json.dumps(dict(event='audited', item_id=item['item_id'], decision=sidecar['decision'],
                          reasons=sidecar.get('review', {}).get('reasons'))), flush=True)
    return sidecar
