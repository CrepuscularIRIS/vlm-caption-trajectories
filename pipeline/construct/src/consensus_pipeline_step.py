"""Advance the consensus-audit pipeline by at most one stage (2026-10-06; run by the 15-min session cron).

Mentor 11: selection first, no new generators, answer agreement only orders the audit, process requirements unchanged,
one suitable complete trajectory per task. Stages (state in <audit>/PIPELINE_STATE.json):
  t2     second tranche audit running -> when done: rebuild pools, build swap round 2, launch its audit
  swap2  swap-2 audit running        -> when done: rebuild pools, export v0.3 (holds), LF check, launch the
                                        fallback re-review of rows held only for a missing fallback verdict
  fbr    fallback re-review running   -> when done: rebuild pools, export v0.3 again, LF check -> done
  done   nothing to do
Each call prints one JSON line (stage, progress, actions). Never kills anything, never generates trajectories.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

SRC = Path(__file__).resolve().parent
RUNS = SRC.parent / 'runs'
AUDIT = RUNS / 'audit_consensus_v1_20261006'
LUNA = RUNS / 'audit_luna_v1_20261003'
FUNNEL = RUNS / 'funnel_v1_20261002'
BENCH = SRC.parent / 'benchmarks'
PY = '${ENV_DIR}/census/bin/python'
LF_PY = '${ENV_DIR}/llamafactory/bin/python'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='')
FROZEN = ['SPOTCHECK_TEMPLATE.json', 'SPOTCHECK_CASEBOOK.html', 'HUMAN_QUEUE.jsonl']
STATE = AUDIT / 'PIPELINE_STATE.json'


def alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text())
        return b'funnel_audit.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except (OSError, ValueError):
        return False


def done_count(log: Path) -> int:
    return sum(1 for l in log.read_text(errors='replace').splitlines() if l.startswith('{"item"')) if log.exists() else 0


def run(args: list[str]) -> str:
    return subprocess.run(args, cwd=SRC, env=ENV, capture_output=True, text=True, check=True).stdout


def rebuild_pools() -> dict:
    """Rebuild POOLS.json; keep the reviewed 192-sample files in place, park the fresh sample for the next round."""
    run([PY, 'build_audit_pools.py', '--audit', str(AUDIT), '--mode', 'multi_generator_consensus', '--reuse',
         str(AUDIT / 'reused.jsonl'), '--reuse-audit', str(LUNA), '--reject-rate', '0.05'])
    nxt = AUDIT / 'spotcheck_next_candidate'
    nxt.mkdir(exist_ok=True)
    for f in FROZEN:
        shutil.copy(AUDIT / f, nxt / f)
        shutil.copy(AUDIT / 'spotcheck_192_frozen' / f, AUDIT / f)
    return json.loads((AUDIT / 'POOLS.json').read_text())['pools']


def launch_audit(manifest: str, log: str, pid: str) -> None:
    cmd = (f"cd {SRC} && CUDA_VISIBLE_DEVICES= nohup setsid {PY} funnel_audit.py --manifest {AUDIT / manifest} --out {AUDIT} "
           f"--sonnet-cap 90000 --gemini-cap 0 --reader grok-4.7 --sonnet-route cc --blind-mode sonnet --escalate-cap 9000 "
           f"--workers 8 > {AUDIT / log} 2>&1 & echo $! > {AUDIT / pid}")
    subprocess.run(['bash', '-c', cmd], check=True)


def export_v03() -> dict:
    out = BENCH / 'sft_v0_3_20261006'
    run([PY, 'build_sft_v0.py', '--audit', str(AUDIT), '--luna-audit', str(LUNA), '--funnel', str(FUNNEL), '--out', str(out),
         '--name', 'caption_sft_v0_3', '--exclude', str(AUDIT / 'SPOTCHECK_EXCLUDE_BAD.txt'),
         str(AUDIT / 'SPOTCHECK_HOLD_FACT_WEAK.txt'), '--include', str(AUDIT / 'SPOTCHECK_ADMIT_KEEP.txt'),
         '--hold-fallback-unreviewed', '--protocol-check'])
    rows = [json.loads(l) for l in (out / 'data.jsonl').open()]
    (out / 'caption_sft_v0_3.json').write_text(json.dumps(rows, ensure_ascii=False))
    info = json.loads((out / 'dataset_info.json').read_text())
    info['caption_sft_v0_3']['file_name'] = 'caption_sft_v0_3.json'
    (out / 'dataset_info.json').write_text(json.dumps(info, indent=1))
    subprocess.run(['bash', '-c', f"cd {SRC} && CUDA_VISIBLE_DEVICES= nohup {LF_PY} lf_preprocess_check.py {out} caption_sft_v0_3 "
                                  f"> {out / 'lf_check.log'} 2>&1 &"], check=True)
    s = json.loads((out / 'STATS.json').read_text())
    return {k: s[k] for k in ('T', 'Q', 'G', 'held', 'held_tasks_without_alternative', 'by_behavior')}


def fallback_manifest() -> int:
    held = [json.loads(l) for l in (BENCH / 'sft_v0_3_20261006' / 'HELD.jsonl').open()]
    keys = [h['key'] for h in held if h['reason'] == 'fallback_unreviewed']
    rows = []
    for k in keys:
        side = AUDIT / 'audit' / f'{k}.json'
        side = side if side.exists() else LUNA / 'audit' / f'{k}.json'
        r = dict(json.loads(side.read_text())['manifest_row'])
        r['tag'] = 'fbr_' + r.get('tag', 'x')
        rows.append(r)
    (AUDIT / 'manifest_fbr.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return len(rows)


def main() -> None:
    st = json.loads(STATE.read_text()) if STATE.exists() else dict(stage='t2')
    out = dict(stage=st['stage'], actions=[])
    if st['stage'] == 't2':
        out['progress'] = f"{done_count(AUDIT / 'audit_t2.log')}/3804"
        if not alive(AUDIT / 'audit_t2.pid'):
            out['pools'] = rebuild_pools()
            run([PY, 'build_swap_round.py', '--funnel', str(FUNNEL), '--audit', str(AUDIT), '--round', '2', '--luna-audit', str(LUNA)])
            out['swap2'] = json.loads((AUDIT / 'SWAP2.json').read_text())
            launch_audit('manifest_swap2.jsonl', 'audit_swap2.log', 'audit_swap2.pid')
            st['stage'] = 'swap2'
            out['actions'].append('t2 done -> pools rebuilt, swap2 audit launched')
    elif st['stage'] == 'swap2':
        n = sum(1 for _ in (AUDIT / 'manifest_swap2.jsonl').open())
        out['progress'] = f"{done_count(AUDIT / 'audit_swap2.log')}/{n}"
        if not alive(AUDIT / 'audit_swap2.pid'):
            out['pools'] = rebuild_pools()
            out['v0_3'] = export_v03()
            out['fbr_rows'] = fallback_manifest()
            if out['fbr_rows']:
                launch_audit('manifest_fbr.jsonl', 'audit_fbr.log', 'audit_fbr.pid')
                st['stage'] = 'fbr'
            else:
                st['stage'] = 'done'
            out['actions'].append('swap2 done -> v0.3 exported, LF check started, fallback re-review launched')
    elif st['stage'] == 'fbr':
        n = sum(1 for _ in (AUDIT / 'manifest_fbr.jsonl').open())
        out['progress'] = f"{done_count(AUDIT / 'audit_fbr.log')}/{n}"
        if not alive(AUDIT / 'audit_fbr.pid'):
            out['pools'] = rebuild_pools()
            out['v0_3'] = export_v03()
            st['stage'] = 'done'
            out['actions'].append('fallback re-review done -> v0.3 re-exported, LF check started')
    else:
        lf = BENCH / 'sft_v0_3_20261006' / 'LF_CHECK.json'
        if lf.exists():
            d = json.loads(lf.read_text())
            out['lf_check'] = {k: d[k] for k in ('samples', 'passed', 'max_tokens')}
    STATE.write_text(json.dumps(st))
    out['next_stage'] = st['stage']
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
