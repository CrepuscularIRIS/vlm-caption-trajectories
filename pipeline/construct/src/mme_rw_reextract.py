"""Verify repaired MME-RealWorld ocr_cc / remote_sensing parts against HF LFS sha256, then re-extract.

1. Fetch expected sha256 (LFS oid) for every <cat>.tar.gz.part_* from the HF tree API.
2. sha256 every local non-.corrupt part; stop the category on any mismatch or missing part.
3. Extract `cat parts | tar xz` into <root>/_reextract_<cat>_<date>/; check both exit codes.
4. Only after a clean extraction: move the old partial directory aside (<cat>.partial_<date>) and move the
   new tree into place. Nothing is deleted. A JSON report is written next to the extraction.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PARTS = Path('${HF_HOME}/bench_downloads/yifanzhang114__MME-RealWorld')
ROOT = Path('${BENCH_DIR}/mme_realworld_full')
TREE = 'https://huggingface.co/api/datasets/yifanzhang114/MME-RealWorld/tree/main'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def expected(cat: str) -> dict:
    rows = requests.get(TREE, timeout=60).json()
    return {r['path']: r['lfs']['oid'] for r in rows if r['path'].startswith(cat + '.tar.gz.part_') and r.get('lfs')}


def run(cat: str, date: str, log) -> dict:
    exp = expected(cat)
    local = {p.name: p for p in PARTS.glob(cat + '.tar.gz.part_*') if not p.name.endswith(('.corrupt', '.redo'))}
    rep = dict(category=cat, expected_parts=len(exp), local_parts=len(local), started=time.time())
    if set(exp) != set(local):
        rep.update(status='part_set_mismatch', missing=sorted(set(exp) - set(local)), extra=sorted(set(local) - set(exp)))
        return rep
    with ThreadPoolExecutor(4) as ex:
        got = dict(zip(sorted(local), ex.map(lambda n: sha(local[n]), sorted(local))))
    bad = [n for n in got if got[n] != exp[n]]
    rep['sha_ok'] = len(got) - len(bad)
    if bad:
        rep.update(status='sha_mismatch', bad=bad)
        return rep
    log(f'{cat}: all {len(got)} parts match HF sha256; extracting')
    dest = ROOT / f'_reextract_{cat}_{date}'
    dest.mkdir(parents=True, exist_ok=False)
    cat_p = subprocess.Popen(['cat', *[str(local[n]) for n in sorted(local)]], stdout=subprocess.PIPE)
    tar_p = subprocess.Popen(['tar', 'xzf', '-', '-C', str(dest)], stdin=cat_p.stdout, stderr=subprocess.PIPE)
    cat_p.stdout.close()
    tar_err = tar_p.communicate()[1].decode(errors='replace')
    rc = (cat_p.wait(), tar_p.returncode)
    rep.update(cat_rc=rc[0], tar_rc=rc[1], tar_stderr=tar_err[-2000:],
               files=sum(1 for p in dest.rglob('*') if p.is_file()))
    if rc != (0, 0):
        rep['status'] = 'extract_failed'
        return rep
    old = ROOT / cat
    if old.exists():
        old.rename(ROOT / f'{cat}.partial_{date}')
    # The archive holds a top-level <cat>/ directory; move that (or the whole tree) into place.
    inner = dest / cat
    shutil.move(str(inner if inner.is_dir() else dest), str(old))
    rep.update(status='ok', installed=str(old), finished=time.time())
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--cats', default='ocr_cc,remote_sensing')
    p.add_argument('--date', default=time.strftime('%Y%m%d'))
    a = p.parse_args()
    out = ROOT / f'REEXTRACT_{a.date}.json'
    log = lambda m: print(time.strftime('%H:%M:%S'), m, flush=True)
    reports = []
    for cat in a.cats.split(','):
        log('start ' + cat)
        reports.append(run(cat, a.date, log))
        log(json.dumps({k: v for k, v in reports[-1].items() if k != 'tar_stderr'}))
        out.write_text(json.dumps(reports, indent=1))
    log('DONE')


if __name__ == '__main__':
    main()
