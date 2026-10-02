"""CPU-only selection for caption v5.1: audit sidecars -> four pools -> LLaMA-Factory export + coverage summary.

Pools (BundleAudit 00 §5.4):
  clean_sft               consensus (>=2 generators correct) and an audited candidate that passed
                          (blind read SUPPORTED, structure review clean); exported for SFT.
  recovery_continuation   passed blind read, final answer correct, but the review found a contradicted
                          claim that a supported REVISE later corrected; stored, not exported.
  calibration_or_ambiguous single-source correct, label suspects (>=2 generators agree on the same
                          non-gold answer), abstentions, blind INSUFFICIENT/CONTRADICTED, review failures.
  counterfactual_diagnostic not produced by this script.
Only records and reasons are written for non-exported pools; raw trajectories stay in the run dirs.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from run_minio3_replica import ROOT


def recovery_like(audit: dict) -> bool:
    p = (audit.get('review') or {}).get('parsed') or {}
    revise_steps = [u['step'] for u in p.get('updates') or [] if u.get('kind') == 'REVISE' and u.get('supported')]
    bad = [c.get('step') for c in p.get('contradicted') or []]
    return bool(bad) and bool(revise_steps) and all(isinstance(b, int) and b < max(revise_steps) for b in bad)


def behaviors(ro: dict) -> dict:
    steps = ro['steps']
    beh = [s.get('parsed', {}).get('behavior') for s in steps]
    upd = [u['kind'] for s in steps for u in s.get('parsed', {}).get('updates', [])]
    return dict(direct=ro['crops'] == 0, fallback='fallback' in beh, hold='HOLD' in upd, revise='REVISE' in upd,
                calc=any(x['kind'] == 'CALC' for s in steps for x in s.get('parsed', {}).get('extra_lines', [])),
                flags=sorted({f for s in steps for f in s.get('flags', [])}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', default='v51_clean')
    p.add_argument('--manual', type=Path, default=None,
                   help='JSON {"item_id::generator": {"verdict": "KEEP|WEAK|BAD|PENDING", "note": "..."}}; when given, '
                        'only KEEP rows are exported and unreviewed passed rows are listed as needs_manual')
    a = p.parse_args()
    a.out, a.audit, a.runs = a.out.resolve(), a.audit.resolve(), [r.resolve() for r in a.runs]
    (a.out / 'sft').mkdir(parents=True, exist_ok=True)
    manual = json.loads(a.manual.read_text()) if a.manual else None
    rows, records, metas = [], [], []
    for sc in sorted((a.audit / 'sidecars').glob('*.json')):
        side = json.loads(sc.read_text())
        iid = side['item_id']
        per = {r.name: json.loads((r / 'results' / f'{iid}.json').read_text()) for r in a.runs
               if (r / 'results' / f'{iid}.json').exists()}
        answers = [per[g]['rollout']['answer'] for g in per if per[g]['rollout']['termination'] == 'answer']
        wrong_agree = [ans for ans, c in Counter(answers).items() if c >= 2
                       and all(per[g]['score'].get('score') == 0 for g in per if per[g]['rollout']['answer'] == ans)]
        passed = next((x for x in side['audited'] if x['passed']), None)
        auto_passed = passed is not None
        verdict = manual.get(f"{iid}::{passed['generator']}", {}).get('verdict') if manual is not None and passed else None
        if passed and manual is not None and verdict in ('WEAK', 'BAD', 'PENDING'):
            pool, reason, passed = 'calibration_or_ambiguous', f'manual_{verdict.lower()}', None
        elif passed:
            pool, reason = 'clean_sft', 'passed'
        elif any(recovery_like(x) and x['blind_read']['answer_agreement'] == 'SUPPORTED' for x in side['audited']):
            pool, reason = 'recovery_continuation', 'contradicted_then_revised'
        elif len(side['correct']) < 2:
            pool = 'calibration_or_ambiguous'
            reason = ('label_suspect' if wrong_agree else 'single_source' if side['correct']
                      else 'all_abstain' if all(per[g]['rollout']['termination'] == 'abstain' for g in per) else 'no_correct')
        else:
            pool = 'calibration_or_ambiguous'
            reason = ';'.join(r for x in side['audited'] for r in x['reasons']) or 'no_clean_candidate'
        row = dict(item_id=iid, split=side['split'], material=side['material'], correct=side['correct'], pool=pool,
                   reason=reason, chosen=passed['generator'] if passed else None, manual=verdict, auto_passed=auto_passed,
                   exportable=bool(passed) and (manual is None or verdict == 'KEEP'))
        if passed:
            row['behaviors'] = behaviors(per[passed['generator']]['rollout'])
        if row['exportable']:
            ro = per[passed['generator']]['rollout']
            rec = dict(ro['export'], images=[str(ROOT / x) if not x.startswith('/') else x for x in ro['export']['images']])
            records.append({k: v for k, v in rec.items() if k != 'protocol'})
            metas.append(dict(item_id=iid, generator=passed['generator'], crops=ro['crops'], budget=ro['budget'],
                              split=side['split'], material=side['material'], protocol=rec['protocol'],
                              export_manifest=ro['export_manifest'], answer_agreement=passed['blind_read']['answer_agreement'],
                              facts=[dict(step=f['step'], claimed=f['claimed'], read=f.get('read'), status=f['status'])
                                     for f in passed.get('facts') or []],
                              reader=passed['blind_read']['model']))
        rows.append(row)
    (a.out / 'selection.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (a.out / 'sft' / f'{a.name}.json').write_text(json.dumps(records, ensure_ascii=False, indent=1))
    (a.out / 'sft' / f'{a.name}_meta.json').write_text(json.dumps(metas, ensure_ascii=False, indent=1))
    (a.out / 'sft' / 'dataset_info.json').write_text(json.dumps({a.name: {
        'file_name': f'{a.name}.json', 'formatting': 'sharegpt', 'columns': {'messages': 'conversations', 'images': 'images'},
        'tags': {'role_tag': 'from', 'content_tag': 'value', 'user_tag': 'human', 'assistant_tag': 'gpt',
                 'system_tag': 'system'}}}, indent=1))
    main_rows = [r for r in rows if r['exportable']]
    auto = [r for r in rows if r['auto_passed']]
    cov = Counter(k for r in main_rows for k, v in r['behaviors'].items() if v is True)
    summ = dict(items=len(rows), pools=dict(Counter(r['pool'] for r in rows)),
                funnel=dict(auto_candidates=len(auto), manual_keep=sum(r['manual'] == 'KEEP' for r in auto),
                            needs_manual=sum(r['manual'] is None for r in auto) if manual is not None else 0,
                            exportable=len(main_rows), manual_gate=manual is not None),
                reasons=dict(Counter(r['reason'] for r in rows if r['pool'] != 'clean_sft')),
                clean_by_material=dict(Counter(r['material'] for r in main_rows)),
                generator_share=dict(Counter(r['chosen'] for r in main_rows)), behavior_coverage=dict(cov),
                clean_with_flags=sum(bool(r['behaviors']['flags']) for r in main_rows),
                smoke_gate=dict(usable=len(main_rows), direct=cov['direct'], hold=cov['hold'], revise=cov['revise'],
                                fallback=cov['fallback']))
    (a.out / 'SUMMARY.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
