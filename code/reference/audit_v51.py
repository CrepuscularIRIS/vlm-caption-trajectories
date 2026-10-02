"""Offline audit for caption v5.1 runs: prefix-limited blind read + one structure review per candidate.

Per item, reading the frozen results of generators A, B (, C):
1. correctness from the deterministic scorer stored in each result (gold never reaches a generator);
2. candidates = correct trajectories without hard violations, ranked: fewer risk flags, fewer crops,
   generator order; at most --max-audit candidates are audited, stopping at the first that passes;
3. blind read (BundleAudit 03 §2): the reader sees the question (with options) and ONLY the views the
   commit step cites, all received before that step; never the caption, the generator's answer or gold.
   Comparator: SUPPORTED / CONTRADICTED / INSUFFICIENT (UNREADABLE);
4. structure review: question + all received views + captions; checks instance switches, flips
   without evidence, HOLD/REVISE validity, withdrawal of dependent conclusions, contradicted claims.
   It does not certify an early claim with later views (that is the blind read's job).
Items where fewer than two generators are correct are not audited (single-source / label-suspect pools).
Writes OUT/sidecars/{item_id}.json; API receipts under OUT/{reader,reviewer}/calls.
"""
import argparse
import json
import logging
import re
from pathlib import Path

import caption_v51 as proto
from corpus_scoring import score
from probe_api import text_msg

LOG = logging.getLogger('audit_v51')
AUDIT_VERSION = 'v51-audit-2'
RISK_FLAGS = {'repeat_crop', 'completion_phrase', 'long_caption', 'no_citation'}
COORD_FLAGS = {'coordinate_talk', 'coordinate_unresolved'}
MAX_FACT_VIEWS = 5
STRONG_READER_MATERIALS = {'text', 'count'}

BLIND_SYSTEM = ('You read images. Use only the images provided. Do not guess from context, brands, typical appearance '
                'or which option sounds plausible. If the images do not show enough to decide, answer UNREADABLE.')
BLIND_USER = ('Question: {question}\nThe image(s) below are the only evidence. Answer the question from them alone.\n'
              'Reply exactly in two lines:\nANSWER: <{fmt} or UNREADABLE>\nEVIDENCE: <one sentence naming the visible detail>')
REVIEW_SYSTEM = 'You audit a step-by-step visual investigation of one image. You do not know the correct answer.'
REVIEW_USER = """Question: {question}

You receive every view the investigator received, in order (labelled), then the investigator's notes per step.
Judge only against the images. Report:
- contradicted: statements about an image that the cited view clearly contradicts;
- instance_switch: steps where the described target is a different object than before and no REVISE line says so;
- unsupported_flip: steps that abandon a supported conclusion without new visual evidence;
- updates: for each HOLD or REVISE line, whether the cited view supports it, and for REVISE whether conclusions that
  depended on the changed claim were withdrawn or recomputed (null if nothing depended on it);
- fallback_real: for each fallback step, whether the abandoned view really lacks what the notes say;
- key_facts: 1 to 3 visual facts the final answer depends on (the reading, count, identity or binding that decides it).
  They must be the underlying observations, never the final answer or the choice between options restated.
  For each: step (the step whose notes first state it), question (a neutral OPEN question about the images that can be
  answered by looking, naming the instance/row/series/legend it concerns, WITHOUT the claimed value and without yes/no
  wording), claimed (ONE short value: a number, colour, word or name, never a sentence), views (ids of views received before that step that show it).
Output only JSON:
{{"contradicted":[{{"step":N,"text":"...","view":"..."}}],"instance_switch":[N],"unsupported_flip":[N],
"updates":[{{"step":N,"kind":"HOLD|REVISE","supported":true,"dependents_withdrawn":true}}],"fallback_real":{{"N":true}},
"key_facts":[{{"step":N,"question":"...","claimed":"...","views":["observation_1"]}}]}}

Notes:
{notes}"""


def decisive_views(ro: dict) -> list[dict]:
    """Views cited by the commit step (minimal evidence set incl. older binding views); last view if none."""
    commit = ro['steps'][-1].get('parsed', {})
    by_id = {v['view_id']: v for v in ro['views']}
    cited = [c for c in commit.get('cited', []) if c in by_id]
    if not cited:
        cited = [ro['views'][-1]['view_id']]
    return sorted((by_id[c] for c in cited), key=lambda v: ro['views'].index(v))[-3:]


FACT_USER = ('Question about the image(s): {question}\nThe image(s) below are the only evidence.\n'
             'Reply exactly in two lines:\nANSWER: <short answer or UNREADABLE>\nEVIDENCE: <one sentence naming the visible detail>')


COMPARE_USER = ('Question: {question}\nCLAIM: {claimed}\nINDEPENDENT READING: {read}\n'
                'Judge only whether the independent reading agrees with the claim. Reply with one word: AGREE, '
                'DISAGREE (the reading contradicts the claim) or UNCLEAR (the reading does not address the claim).')


def parse_answer_line(text: str) -> str | None:
    m = re.search(r'ANSWER\s*:\s*(.+)', text or '')
    return m.group(1).strip() if m else None


def compare(spec_kind: str, traj_answer: str, blind: str | None, n_options: int | None) -> str:
    if blind is None:
        return 'UNPARSED'
    if 'UNREADABLE' in blind.upper():
        return 'INSUFFICIENT'
    kind = 'letter' if spec_kind == 'letter' else 'short'
    ok = score({'kind': kind, 'gold': traj_answer, 'n_options': n_options}, blind)['score']
    return 'SUPPORTED' if ok == 1 else 'CONTRADICTED'


def first_json(text: str) -> dict | None:
    m = re.search(r'\{.*\}', text or '', re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def image_block(api, views: list[dict]) -> list:
    out = []
    for v in views:
        out += [{'type': 'text', 'text': f"[{v['view_id']}]"}, api.image(v['path'])]
    return out


def blind_read(api, item: dict, ro: dict) -> dict:
    views = decisive_views(ro)
    fmt = 'one option letter' if item['answer_spec']['kind'] == 'letter' else 'a short answer'
    msgs = [text_msg('system', BLIND_SYSTEM),
            {'role': 'user', 'content': [{'type': 'text', 'text': BLIND_USER.format(question=item['question'], fmt=fmt)}]
             + image_block(api, views)}]
    r = api.call(f"{item['item_id']}/blind", msgs, 2048, replicate_id=item['item_id'] + '#blind', temperature=0.0)
    ans = parse_answer_line(r.get('content'))
    agreement = compare(item['answer_spec']['kind'], ro['answer'], ans, item['answer_spec'].get('n_options'))
    return dict(model=api.model, views=[v['view_id'] for v in views], raw=r.get('content'), answer=ans,
                answer_agreement=agreement, request_sha256=r.get('request_sha256'), usage=r.get('usage'))


def review(api, item: dict, ro: dict) -> dict:
    notes = '\n'.join(f"Step {s['step']} ({s.get('parsed', {}).get('behavior')}): {s.get('parsed', {}).get('caption', '')}"
                      for s in ro['steps'])
    msgs = [text_msg('system', REVIEW_SYSTEM),
            {'role': 'user', 'content': image_block(api, ro['views'])
             + [{'type': 'text', 'text': REVIEW_USER.format(question=item['question'], notes=notes)}]}]
    r = api.call(f"{item['item_id']}/review", msgs, 2048, replicate_id=item['item_id'] + '#review', temperature=0.0)
    return dict(model=api.model, raw=r.get('content'), parsed=first_json(r.get('content')),
                request_sha256=r.get('request_sha256'), usage=r.get('usage'))


def _fact_ok(f) -> bool:
    return (isinstance(f, dict) and isinstance(f.get('step'), int) and isinstance(f.get('question'), str)
            and isinstance(f.get('claimed'), str) and f['claimed'].strip() and isinstance(f.get('views'), list)
            and all(isinstance(v, str) for v in f['views']))


def schema_valid(p) -> bool:
    return (isinstance(p, dict) and isinstance(p.get('contradicted'), list) and isinstance(p.get('instance_switch'), list)
            and isinstance(p.get('unsupported_flip'), list) and isinstance(p.get('fallback_real'), dict)
            and isinstance(p.get('updates'), list) and all(isinstance(u, dict) for u in p['updates'])
            and isinstance(p.get('key_facts'), list) and 1 <= len(p['key_facts']) <= 3
            and all(_fact_ok(f) for f in p['key_facts']))


def review_passes(rv: dict) -> tuple[bool, list[str]]:
    p = rv.get('parsed')
    if p is None:
        return False, ['review_unparsed']
    if not schema_valid(p):
        return False, ['review_schema_invalid']
    why = []
    if p['contradicted']:
        why.append('contradicted_claim')
    if p['instance_switch']:
        why.append('instance_switch')
    if p['unsupported_flip']:
        why.append('unsupported_flip')
    for u in p['updates']:
        if u.get('supported') is False:
            why.append(f"unsupported_{str(u.get('kind')).lower()}")
        if u.get('kind') == 'REVISE' and u.get('dependents_withdrawn') is False:
            why.append('revise_dependents_kept')
    if any(v is False for v in p['fallback_real'].values()):
        why.append('fallback_not_real')
    return not why, why


def check_fact(api, item: dict, ro: dict, fact: dict) -> dict:
    """Neutral blind read of one key fact from the views the investigator had BEFORE the step that states it."""
    n = fact['step']
    out = dict(step=n, question=fact['question'], claimed=fact['claimed'])
    if not 1 <= n <= len(ro['steps']):
        return dict(out, status='NO_VIEW', views=[])
    known = {v['view_id']: v for v in ro['views']}
    allowed = ro['steps'][n - 1]['available_before']
    views = [v for v in dict.fromkeys(fact['views']) if v in allowed and v in known]
    if len(fact['views']) > MAX_FACT_VIEWS:
        return dict(out, status='TOO_MANY_VIEWS', views=views)
    if not views:
        return dict(out, status='NO_VIEW', views=[])
    msgs = [text_msg('system', BLIND_SYSTEM),
            {'role': 'user', 'content': [{'type': 'text', 'text': FACT_USER.format(question=fact['question'])}]
             + image_block(api, [known[v] for v in views])}]
    r = api.call(f"{item['item_id']}/fact{n}", msgs, 2048, replicate_id=f"{item['item_id']}#fact{n}", temperature=0.0)
    read = parse_answer_line(r.get('content'))
    if read is None:
        status = 'UNPARSED'
    elif 'UNREADABLE' in read.upper():
        status = 'INSUFFICIENT'
    elif score({'kind': 'short', 'gold': fact['claimed']}, read)['score'] == 1:
        status = 'SUPPORTED'
    else:   # the reading is frozen; only now is the claim shown, to a text-only comparison
        cr = api.call(f"{item['item_id']}/fact{n}cmp", [text_msg('user', COMPARE_USER.format(
            question=fact['question'], claimed=fact['claimed'], read=read))], 512,
            replicate_id=f"{item['item_id']}#fact{n}cmp", temperature=0.0)
        word = (re.search(r'AGREE|DISAGREE|UNCLEAR', (cr.get('content') or '').upper()) or [None])[0]
        status = {'AGREE': 'SUPPORTED', 'DISAGREE': 'CONTRADICTED', 'UNCLEAR': 'INSUFFICIENT'}.get(word, 'UNPARSED')
    return dict(out, status=status, views=views, read=read, model=api.model, raw=r.get('content'),
                request_sha256=r.get('request_sha256'), usage=r.get('usage'))


def rare_behavior(ro: dict) -> bool:
    beh = [s.get('parsed', {}).get('behavior') for s in ro['steps']]
    upd = [u['kind'] for s in ro['steps'] for u in s.get('parsed', {}).get('updates', [])]
    return 'fallback' in beh or 'HOLD' in upd or 'REVISE' in upd


def coord_flags(ro: dict) -> set[str]:
    """Recomputed from stored captions, so results written under an older detector are re-judged."""
    return {f for s in ro['steps'] for f in proto.soft_flags(s.get('parsed', {})) if f in COORD_FLAGS}


def _eligible(per: dict, g: str) -> bool:
    return bool(per.get(g) and per[g]['score'].get('score') == 1 and per[g]['rollout']['termination'] == 'answer'
                and not any(s.get('violations') for s in per[g]['rollout']['steps']))


def excluded(per: dict, gens: list[str]) -> dict[str, list[str]]:
    return {g: sorted(coord_flags(per[g]['rollout'])) for g in gens if _eligible(per, g) and coord_flags(per[g]['rollout'])}


def candidates(per: dict, gens: list[str]) -> list[str]:
    """Eligible trajectories, rare behaviours first, then fewer risk flags, fewer crops, generator order."""
    ok = [g for g in gens if _eligible(per, g) and not coord_flags(per[g]['rollout'])]
    risk = lambda g: sum(f in RISK_FLAGS for s in per[g]['rollout']['steps'] for f in s.get('flags', []))
    return sorted(ok, key=lambda g: (not rare_behavior(per[g]['rollout']), risk(g), per[g]['rollout']['crops'], gens.index(g)))


def audit_candidate(item: dict, ro: dict, g: str, reader, reviewer_api) -> dict:
    br = blind_read(reader, item, ro)
    rv = facts = None
    if br['answer_agreement'] != 'SUPPORTED':
        return dict(generator=g, blind_read=br, review=None, facts=None, passed=False,
                    reasons=[f"answer_{br['answer_agreement'].lower()}"])
    rv = review(reviewer_api, item, ro)
    ok, why = review_passes(rv)
    if ok:
        facts = [check_fact(reader, item, ro, f) for f in rv['parsed']['key_facts']]
        why = [f"fact_{f['status'].lower()}" for f in facts if f['status'] != 'SUPPORTED']
        ok = not why
    return dict(generator=g, blind_read=br, review=rv, facts=facts, passed=ok, reasons=why)


def main() -> None:
    from trajgen_models import make_api
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, required=True, help='generator run dirs, priority order')
    p.add_argument('--items', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reader', default='claude-sonnet-5-5')
    p.add_argument('--reader-strong', default='gpt-6.1-sol')
    p.add_argument('--reviewer', default='claude-sonnet-5-5')
    p.add_argument('--max-audit', type=int, default=2)
    a = p.parse_args()
    a.out, a.runs = a.out.resolve(), [r.resolve() for r in a.runs]
    (a.out / 'sidecars').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    readers = {m: make_api(m, a.out / 'reader', max_tokens=2048, effort=None) for m in {a.reader, a.reader_strong}}
    reviewer = make_api(a.reviewer, a.out / 'reviewer', max_tokens=2048, effort=None)
    strong_reviewer = make_api(a.reader_strong, a.out / 'reviewer_strong', max_tokens=2048, effort=None)
    gens = [r.name for r in a.runs]
    items = {json.loads(l)['item_id']: json.loads(l) for l in a.items.read_text().splitlines()}
    ids = sorted({f.stem for r in a.runs for f in (r / 'results').glob('*.json')} & set(items))
    for iid in ids:
        dest = a.out / 'sidecars' / f'{iid}.json'
        if dest.exists():
            continue
        item = items[iid]
        per = {r.name: json.loads((r / 'results' / f'{iid}.json').read_text()) for r in a.runs
               if (r / 'results' / f'{iid}.json').exists()}
        correct = [g for g in gens if per.get(g) and per[g]['score'].get('score') == 1]
        order = candidates(per, gens)
        side = dict(item_id=iid, split=item['split'], material=item['material'], generators=gens, correct=correct,
                    answers={g: per[g]['rollout']['answer'] for g in per}, audit_version=AUDIT_VERSION,
                    candidate_order=order, excluded=excluded(per, gens), audited=[])
        if len(correct) >= 2:
            for g in order[:a.max_audit]:
                own = per[g]['model']
                reader = readers[a.reader_strong if item['material'] in STRONG_READER_MATERIALS else a.reader]
                if reader.model == own:
                    reader = readers[a.reader_strong]
                res = audit_candidate(item, per[g]['rollout'], g, reader, reviewer if reviewer.model != own else strong_reviewer)
                side['audited'].append(res)
                LOG.info('%s %s answer=%s passed=%s %s', iid, g, res['blind_read']['answer_agreement'], res['passed'], res['reasons'])
                if res['passed']:
                    break
        dest.write_text(json.dumps(side, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
