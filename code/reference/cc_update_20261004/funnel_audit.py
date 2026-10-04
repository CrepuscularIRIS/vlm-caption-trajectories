"""Funnel semantic audit: Sonnet (official cc) reviews structure/binding and proposes neutral key facts;
Gemini (agy) blind-reads each fact from ONLY the views available before the claiming step, without the
question options, notes, claimed value or gold; the frozen reading is then compared with the claim by Sonnet in a
separate text-only process. A final blind read of the commit's cited views is done by Gemini too.

Reuses the v5.1 prompts (audit_v51) and the strict structure checks (audit_tiered). Decisions:
  AUTO_PASS_PENDING_MANUAL   all checks passed; still needs a human KEEP before export
  REVIEW_REJECTED            reasons listed (contradicted, instance switch, unsupported flip, fact not supported ...)
  TRANSPORT_INVALID          a CLI receipt failed validation (wrong model, image not opened, extra tool ...)
  PENDING_AUDIT_BUDGET       budget cap reached before finishing
Inputs: --manifest jsonl rows {result_path, item_id?, tag?}; outputs one sidecar per row in --out/audit/.
Budgets are durable (tiered_transport.Ledger) and count every CLI invocation, including retries.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

from agy_reader import AgyReader
from api_reader import ApiReader
from audit_tiered import cpu_reasons, fact_views, independent_compare, strict_review
from audit_v51 import BLIND_SYSTEM, BLIND_USER, FACT_USER, REVIEW_SYSTEM, REVIEW_USER, first_json, parse_answer_line
from funnel_select import FAMILY
from cc_proxy_reviewer import CCProxyReviewer
from tiered_transport import BudgetStop, CCReviewer, Ledger, write_json


def fs_family(model: str) -> str:
    return FAMILY.get(model, model)

SONNET, GEMINI = 'claude-sonnet-5-5', 'gemini-3.8-flash-low'
READER_FAMILY = {'gemini-3.8-flash-low': 'google', 'grok-4.7': 'xai', 'gpt-6.1-sol': 'openai'}
# Calibration 2026-10-03 (25 manually judged smoke trajectories): exact-shade comparisons rejected a KEEP
# ('dark gray' vs 'Black'); compare on what matters for the question instead.
COMPARE_SYSTEM = 'Compare a frozen independent reading with a short claim. No images or outside knowledge.'
COMPARE_USER2 = ('Question: {question}\nCLAIM: {claimed}\nINDEPENDENT READING: {read}\n'
                 'Reply with one word. AGREE if the reading matches the claim or differs only in wording or a near shade '
                 '(e.g. dark gray vs black) that does not change which object/option the claim identifies, or if the reading '
                 'is a range or approximate count that includes the claimed number (e.g. "about 20 to 25" for 25). DISAGREE if the '
                 'reading gives a different value, count, identity or text. UNCLEAR if the reading does not address the claim.')
REWRITABLE = {'leading_yes_no_fact', 'future_or_unknown_fact_view', 'invalid_key_fact_question_or_value',
              'missing_or_invalid_fallback_review', 'missing_or_spurious_update_review'}
YESNO_FIX = ('\nYour previous JSON was incomplete or invalid: every HOLD/REVISE line needs an "updates" entry and every '
             'step labelled fallback needs a "fallback_real" entry; key facts must not be yes/no questions, must not contain the '
             'value, and may only use views the investigator had already received at that step. Rewrite every key fact: an OPEN question that names the instance '
             'and asks for the value (e.g. "What colour is ...", "How many ..."), and views only from that step\'s '
             '"available before" list. Output the full JSON again.')
REVIEW_EXTRA = ('\nPrefer one decisive instance-attribute binding fact; use two or three only if genuinely needed. '
                'Report every HOLD/REVISE line and every fallback. Never support an early statement with a later view. '
                'The key-fact question must be open and must not contain its claimed value or offer suggested answers. '
                'Do not excuse guessed detail just because the final answer looks plausible.')


def compare(sonnet, label, claimed, read, question, kind='short', n_options=None) -> str:
    if kind == 'letter' or read is None or 'UNREADABLE' in (read or '').upper():
        return independent_compare(sonnet, label, claimed, read, question, kind, n_options)
    from corpus_scoring import score
    if score(dict(kind='short', gold=claimed), read)['score'] == 1:
        return 'SUPPORTED'
    r = sonnet.call(label + '/compare2', COMPARE_SYSTEM, COMPARE_USER2.format(question=question, claimed=claimed, read=read), [])
    if not r['valid']:
        return 'INVALID_RECEIPT'
    word = r['text'].strip().strip('`.').upper()
    return {'AGREE': 'SUPPORTED', 'DISAGREE': 'CONTRADICTED', 'UNCLEAR': 'INSUFFICIENT'}.get(word, 'UNPARSED')


def resolve_views(result: dict, result_path: str) -> list[str]:
    """A view file overwritten/removed by an older redo is restored from <model dir>/media/<sha256>.png (same bytes).
    Returns the ids that could not be resolved."""
    media = Path(result_path).parent.parent / 'media'
    missing = []
    for v in result['rollout']['views']:
        if Path(v['path']).exists():
            continue
        alt = media / f"{v.get('sha256', '')}.png"
        if v.get('sha256') and alt.exists():
            v['path'], v['restored_from_media'] = str(alt), True
        else:
            missing.append(v['view_id'])
    return missing


def audit(result: dict, tag: str, sonnet: CCReviewer, gemini: AgyReader) -> dict:
    item, ro = result['item'], result['rollout']
    if result['model'].startswith('claude'):
        raise ValueError('A reviewer family cannot audit its own generation')
    if fs_family(result['model']) == READER_FAMILY.get(gemini.model):
        raise ValueError('The blind reader is from the generator family')
    prefix = f"{tag}/{item['item_id']}/{result['model']}"
    out = dict(item_id=item['item_id'], model=result['model'], tag=tag, cpu=cpu_reasons(result), facts=[], reasons=[],
               process_reviewer=SONNET + (' via cc' if isinstance(sonnet, CCProxyReviewer) else ''), blind_reader=getattr(gemini, 'model', None))
    if out['cpu']:
        out.update(decision='CPU_REJECTED', reasons=out['cpu'])
        return out
    notes = '\n'.join(f"Step {s['step']} (available before: {', '.join(s['available_before'])}): {s['parsed']['caption']}"
                      for s in ro['steps'])
    sr = sonnet.call(prefix + '/structure', REVIEW_SYSTEM, REVIEW_USER.format(question=item['question'], notes=notes)
                     + REVIEW_EXTRA, ro['views'])
    if not sr['valid']:  # one technical retry (e.g. not every view opened), new label so the ledger counts it
        sr = sonnet.call(prefix + '/structure_retry', REVIEW_SYSTEM, REVIEW_USER.format(question=item['question'], notes=notes)
                         + REVIEW_EXTRA, ro['views'])
        out['structure_retried'] = True
    out.update(structure_receipt=sr['request_sha256'], structure=first_json(sr.get('text')))
    if not sr['valid']:
        out.update(decision='TRANSPORT_INVALID', reasons=['structure_receipt_invalid'])
        return out
    out['reasons'] = strict_review(out['structure'], ro)
    if out['reasons'] and set(out['reasons']) <= REWRITABLE:  # reviewer-side schema problems: one rewrite, not a rejection
        sr = sonnet.call(prefix + '/structure_rewrite', REVIEW_SYSTEM, REVIEW_USER.format(question=item['question'], notes=notes)
                         + REVIEW_EXTRA + YESNO_FIX, ro['views'])
        out.update(structure_receipt=sr['request_sha256'], structure=first_json(sr.get('text')), structure_rewritten=True)
        if not sr['valid']:
            out.update(decision='TRANSPORT_INVALID', reasons=['structure_receipt_invalid'])
            return out
        out['reasons'] = strict_review(out['structure'], ro)
    # Reviewer omissions are not trajectory faults (P1: 155 rejections were only these, even after one rewrite).
    # A missing fallback verdict is noted (self-declared 'fallback' words are not trusted anyway); a missing HOLD/REVISE
    # verdict sends an otherwise clean trajectory to a human instead of auto-pass. Fact checks still run.
    if 'missing_or_invalid_fallback_review' in out['reasons']:
        out['reasons'].remove('missing_or_invalid_fallback_review')
        out['fallback_unreviewed'] = True
    if 'missing_or_spurious_update_review' in out['reasons']:
        out['reasons'].remove('missing_or_spurious_update_review')
        out['update_unreviewed'] = True
    if out['reasons']:
        out['decision'] = 'REVIEW_REJECTED'
        return out
    for k, fact in enumerate(out['structure']['key_facts']):
        label = prefix + f'/fact{k}'
        # Gemini sees only the prefix views and the neutral question: no options, notes, claimed value or gold.
        br = gemini.call(label, BLIND_SYSTEM, FACT_USER.format(question=fact['question']), fact_views(ro, fact))
        read = parse_answer_line(br.get('text'))
        status = compare(sonnet, label, fact['claimed'], read, fact['question']) if br['valid'] else 'INVALID_RECEIPT'
        entry = dict(fact=fact, read=read, status=status, receipt=br['request_sha256'])
        if status in ('CONTRADICTED', 'INSUFFICIENT'):
            # One independent second read by the other reviewer family; agreement with the claim -> human decides.
            sr2 = sonnet.call(label + '/second_read', BLIND_SYSTEM, FACT_USER.format(question=fact['question']), fact_views(ro, fact))
            read2 = parse_answer_line(sr2.get('text'))
            status2 = compare(sonnet, label + '/second', fact['claimed'], read2, fact['question']) if sr2['valid'] else 'INVALID_RECEIPT'
            entry.update(second_read=read2, second_status=status2, second_receipt=sr2['request_sha256'])
            if status2 == 'SUPPORTED':
                status = entry['status'] = 'DISPUTED'
        out['facts'].append(entry)
        if status != 'SUPPORTED':
            out['reasons'].append('key_fact_' + status.lower())
    commit = ro['steps'][-1]
    ids = commit['parsed'].get('cited', []) or [ro['views'][-1]['view_id']]
    views = fact_views(ro, dict(step=commit['step'], views=ids))
    kind = item['answer_spec']['kind']
    fmt = 'one option letter' if kind == 'letter' else 'a short answer'
    br = gemini.call(prefix + '/final_blind', BLIND_SYSTEM, BLIND_USER.format(question=item['question'], fmt=fmt), views)
    read = parse_answer_line(br.get('text'))
    agreement = (independent_compare(sonnet, prefix + '/final_blind', ro['answer'], read, item['question'], kind,
                                     item['answer_spec'].get('n_options')) if br['valid'] else 'INVALID_RECEIPT')
    out['final_blind'] = dict(read=read, status=agreement, receipt=br['request_sha256'], views=ids)
    if agreement in ('CONTRADICTED', 'INSUFFICIENT'):  # same one-time second read by the other family
        sr2 = sonnet.call(prefix + '/final_second', BLIND_SYSTEM, BLIND_USER.format(question=item['question'], fmt=fmt), views)
        read2 = parse_answer_line(sr2.get('text'))
        status2 = (independent_compare(sonnet, prefix + '/final_second', ro['answer'], read2, item['question'], kind,
                                       item['answer_spec'].get('n_options')) if sr2['valid'] else 'INVALID_RECEIPT')
        out['final_blind'].update(second_read=read2, second_status=status2, second_receipt=sr2['request_sha256'])
        if status2 == 'SUPPORTED':
            agreement = out['final_blind']['status'] = 'DISPUTED'
    if agreement != 'SUPPORTED':
        out['reasons'].append('final_blind_' + agreement.lower())
    if any(r.endswith('invalid_receipt') for r in out['reasons']):
        out['decision'] = 'TRANSPORT_INVALID'
    elif out['reasons'] and all(r.endswith('_disputed') for r in out['reasons']):
        out['decision'] = 'DISPUTED_PENDING_MANUAL'
    elif not out['reasons'] and out.get('update_unreviewed'):
        out['decision'] = 'DISPUTED_PENDING_MANUAL'
        out['reasons'] = ['update_unreviewed']
    else:
        out['decision'] = 'AUTO_PASS_PENDING_MANUAL' if not out['reasons'] else 'REVIEW_REJECTED'
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sonnet-cap', type=int, required=True)
    p.add_argument('--gemini-cap', type=int, required=True, help='cap for the blind reader (Gemini or replacement)')
    p.add_argument('--reader', default=GEMINI, choices=sorted(READER_FAMILY),
                   help='blind fact reader: Gemini via agy, or Grok / Sol via API')
    p.add_argument('--sonnet-route', default='official', choices=('official', 'cc'),
                   help='official = task-local official claude; cc = user cc (claude-kimi, CLIProxy OAuth pool)')
    p.add_argument('--workers', type=int, default=1, help='trajectories audited in parallel (each runs its CLI calls in order)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(a.out / 'budget', {SONNET: a.sonnet_cap, 'cc-' + SONNET: a.sonnet_cap, a.reader: a.gemini_cap})
    sonnet = (CCProxyReviewer(a.out / 'sonnet_cc', ledger, SONNET) if a.sonnet_route == 'cc'
              else CCReviewer(a.out / 'sonnet', ledger, SONNET))
    gemini = (AgyReader(a.out / 'gemini', ledger, GEMINI) if a.reader == GEMINI
              else ApiReader(a.out / a.reader.split('-')[0], ledger, a.reader))
    rows = [json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    decisions = Counter()
    lock = threading.Lock()

    def one(row):
        result = json.loads(Path(row['result_path']).read_text())
        dest = a.out / 'audit' / f"{row.get('tag', 'x')}__{result['item']['item_id']}__{result['model']}.json"
        if dest.exists():
            return json.loads(dest.read_text())['decision']
        missing = resolve_views(result, row['result_path'])
        try:
            if missing:
                raise FileNotFoundError('views not recoverable: ' + ', '.join(missing))
            side = audit(result, row.get('tag', 'x'), sonnet, gemini)
        except BudgetStop as exc:
            side = dict(item_id=result['item']['item_id'], model=result['model'], decision='PENDING_AUDIT_BUDGET', reason=str(exc))
        except ValueError as exc:  # e.g. a Claude/Gemini-generated trajectory: no independent reviewer family left
            side = dict(item_id=result['item']['item_id'], model=result['model'], decision='SKIPPED', reason=str(exc))
        except Exception as exc:  # noqa: BLE001 -- one broken record must not stop the batch; logged in its sidecar
            side = dict(item_id=result['item']['item_id'], model=result['model'], decision='AUDIT_ERROR',
                        reason=f'{type(exc).__name__}: {exc}'[:300])
        side.update(result_path=row['result_path'], manifest_row=row)
        write_json(dest, side)
        with lock:
            print(json.dumps(dict(item=side['item_id'], model=side['model'], decision=side['decision'],
                                  reasons=side.get('reasons'))), flush=True)
        return side['decision']

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for d in ex.map(one, rows):
            decisions[d] += 1
    write_json(a.out / 'AUDIT_SUMMARY.json', dict(rows=len(rows), decisions=dict(decisions),
                                                   dispatch=dict(Counter(r['model'] for r in ledger.rows()))))
    print(json.dumps(dict(decisions)))


if __name__ == '__main__':
    main()
