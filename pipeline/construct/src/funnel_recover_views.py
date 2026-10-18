"""Re-link archived funnel attempts to their original crop bytes (one-off repair, append-only).

Before 2026-10-03 a redo reused obs/<item_id>/, overwriting the crop images that the archived attempt0
result cites. The bytes survive in <model dir>/media/<sha256>.png. This script verifies each entry of the
Codex recovery manifest by hashing the media file and writes VIEW_RECOVERY.jsonl in the run root:
{result, view, expected_sha256, path}. Archived result JSON files are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(l) for l in a.manifest.read_text().splitlines() if l.strip()]
    ok, bad = [], []
    for r in rows:
        media = Path(r['recoverable_media'])
        if media.exists() and sha(media) == r['expected']:
            ok.append(dict(result=r['result'], view=r['view'], expected_sha256=r['expected'], path=str(media)))
        else:
            bad.append(r)
    out = a.root / 'VIEW_RECOVERY.jsonl'
    out.write_text(''.join(json.dumps(x) + '\n' for x in ok))
    print(json.dumps(dict(entries=len(rows), verified=len(ok), unrecoverable=len(bad),
                          results=len({x['result'] for x in ok}), written=str(out))))


if __name__ == '__main__':
    main()
