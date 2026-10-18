"""Export the 4,000-task selection to LLaMA-Factory sharegpt (2026-10-07). CPU only, no rewriting.

Each row uses the rollout's own export (teacher scaffold already removed at generation). Image paths are made absolute;
a missing view is restored from <model dir>/media/<sha256>.png when possible, else the task is reported (not silently
dropped). The protocol check (export_protocol_check.HOLD) is re-applied; held rows are reported. Writes
<out>/caption_sft_4k.json, dataset_info.json, MANIFEST.jsonl, PROBLEMS.jsonl, STATS.json.
"""
import argparse
import collections
import json
from pathlib import Path

from export_protocol_check import HOLD, check_record

ROOT = Path('${ICML_DIR}')
NAME = 'caption_sft_4k'


def image_paths(rec: dict, ro: dict, result_path: Path) -> tuple[list[str], list[str]]:
    by_sha = {Path(v['path']).name: v.get('sha256') for v in ro['views']}
    out, missing = [], []
    for x in rec['images']:
        p = Path(x) if Path(x).is_absolute() else ROOT / x
        if not p.exists():
            alt = result_path.parent.parent / 'media' / f"{by_sha.get(p.name, '')}.png"
            if alt.exists():
                p = alt
            else:
                missing.append(str(p))
        out.append(str(p))
    return out, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, manifest, problems = [], [], []
    for line in a.selection.open():
        s = json.loads(line)
        rp = Path(s['result_path'])
        ro = json.loads(rp.read_text())['rollout']
        rec = ro['export']
        bad = sorted(set(check_record(rec)) & HOLD)
        imgs, missing = image_paths(rec, ro, rp)
        if bad or missing:
            problems.append(dict(item_id=s['item_id'], result_path=str(rp), protocol=bad, missing_images=missing[:3]))
            continue
        rows.append(dict(conversations=rec['conversations'], images=imgs))
        manifest.append(dict(s, status='model_reviewed_not_human_accepted', n_images=len(imgs)))
    (a.out / f'{NAME}.json').write_text(json.dumps(rows, ensure_ascii=False))
    (a.out / 'dataset_info.json').write_text(json.dumps({NAME: dict(file_name=f'{NAME}.json', formatting='sharegpt',
        columns=dict(messages='conversations', images='images'),
        tags=dict(role_tag='from', content_tag='value', user_tag='human', assistant_tag='gpt', system_tag='system'))}, indent=1))
    for fn, rs in (('MANIFEST.jsonl', manifest), ('PROBLEMS.jsonl', problems)):
        (a.out / fn).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rs))
    C = collections.Counter
    st = dict(T=len(manifest), Q=len({m['item_id'] for m in manifest}), G=len({m['group_id'] for m in manifest}), problems=len(problems),
              by_bench=dict(C(m['bench'] for m in manifest)), by_source=dict(C(m['source'] for m in manifest)),
              by_behavior=dict(C(m['behavior'] for m in manifest)), by_model=dict(C(m['model'] for m in manifest)),
              status='model_reviewed_not_human_accepted')
    (a.out / 'STATS.json').write_text(json.dumps(st, indent=1, ensure_ascii=False))
    print(json.dumps(st, ensure_ascii=False))


if __name__ == '__main__':
    main()
