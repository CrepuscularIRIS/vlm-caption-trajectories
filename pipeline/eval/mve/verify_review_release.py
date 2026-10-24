"""CPU-only structural and numerical verification of the reviewed evidence release.

Does not assess visual truth, causal validity, or training efficacy.
Use --write-manifest only after deliberately changing release artifacts.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/review_20261022"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (int, float)):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    records = [json.loads(line) for line in (DATA / "reviewed_trajectories.jsonl").open()]
    ids = {r["id"] for r in records}
    assert ids == {f"{p}{i:02}" for p in "su" for i in range(40)}
    assert len(records) == 80
    byid = {r["id"]: r for r in records}
    for cohort in ["receipt", "utilization"]:
        rows = list(csv.DictReader((DATA / cohort / "annotations.csv").open()))
        assert len(rows) == 40
        for r in rows:
            assert byid[r["id"]]["annotation"] == r
    for r in records:
        assert r["turns"] and len(r["turns"]) == r["trajectory"]["n_crops"] + 1
        assert all("text" in t for t in r["turns"])
        assert re.fullmatch(r"[0-9a-f]{64}", r["source_sha256"])
        assert r["reviewed_views"]
        assert all(re.fullmatch(r"[0-9a-f]{64}", v["sha256"]) for v in r["reviewed_views"])
        assert r["image_bytes_in_release"] is True
        for v in r["reviewed_views"]:
            p = ROOT / v["public_image"]
            assert p.stat().st_size == v["bytes"] and sha(p) == v["sha256"], p
            assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", p
        assert not {"question", "images", "system", "prompt"} & r.keys()
    assert sum(len(r["reviewed_views"]) for r in records if r["cohort"] == "utilization") == 129
    assert sum(len(r["reviewed_views"]) for r in records if r["cohort"] == "receipt") == 147
    images = {ROOT / v["public_image"] for r in records for v in r["reviewed_views"]}
    assert len(images) == 274
    anchors = load(DATA / "segment_boundaries.json")
    assert len(anchors) == 13
    assert collections.Counter(a["group"] for a in anchors) == {"core": 9, "mixed": 4}
    for a in anchors:
        assert "question" not in a
        for label, turn, span in [("fact", a["fact_turn"], a["fact_char_range_in_rendered_turn"]),
                                  ("error", a["decisive_error_turn"], a["error_char_range_in_rendered_turn"])]:
            text = byid[a["id"]]["turns"][turn - 1]["text"]
            assert text[span[0]:span[1]] == a[label + "_verbatim"], a["id"]
    logs = ROOT / "logs/mve/segval_v1"
    runs = [json.loads(line) for p in sorted(logs.glob("runs_*.jsonl")) for line in p.open()]
    assert len(runs) == 832
    assert len({(r["id"], r["prefix"], r["j"]) for r in runs}) == 832
    cells = collections.defaultdict(list)
    for r in runs:
        assert r["y"] in [0, 1] and r["crops"] >= 0 and r["gen_tokens"] >= 0
        assert math.isclose(r["r"], r["y"] * (1 - .01 * r["crops"]), abs_tol=1e-12)
        cells[r["id"], r["prefix"]].append(r)
    assert len(cells) == 52 and all(len(v) == 16 for v in cells.values())
    saved = load(ROOT / "logs/mve/segval_summary.json")
    for (cid, prefix), rs in cells.items():
        assert set(r["j"] for r in rs) == set(range(16))
        assert sum(r["y"] for r in rs) / 16 == saved["per_case"][cid][prefix]["V"]
        same(byid[cid]["v1_measurement"], saved["per_case"][cid])
    with tempfile.TemporaryDirectory() as td:
        for path in logs.glob("runs_*.jsonl"):
            shutil.copyfile(path, Path(td) / path.name)
        subprocess.run([sys.executable, str(ROOT / "pipeline/eval/mve/summarize_segment.py"), td],
                       check=True, stdout=subprocess.DEVNULL)
        same(load(Path(td) / "summary.json"), saved)
    files = sorted([p for p in DATA.rglob("*") if p.is_file() and p.name != "release_manifest.json"]
                   + list((ROOT / "pipeline/eval/mve").glob("*.py"))
                   + [ROOT / "pipeline/eval/mve/README.md", ROOT / "logs/mve/segval_summary.json",
                      ROOT / "docs/archive/30_rl_credit_report_20261022.md"]
                   + [p for p in logs.iterdir() if p.is_file()] + list(images))
    # Package contents must not include machine endpoints or source prompts.
    forbidden = ["/home/lingxufeng/", "/root/autodl-tmp/", "connect.westd.seetacloud.com", "-----BEGIN PRIVATE KEY-----"]
    for p in files:
        if p.suffix == ".png":
            continue
        text = p.read_text()
        assert not any(s in text for s in forbidden if p != Path(__file__).resolve()), p
        if p.suffix == ".md":
            for dest in re.findall(r"\]\(([^)]+)\)", text):
                if re.match(r"[a-z]+://", dest) or dest.startswith("#"):
                    continue
                assert (p.parent / dest.split("#")[0]).exists(), (p, dest)
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in files}
    mp = DATA / "release_manifest.json"
    if args.write_manifest:
        mp.write_text(json.dumps(manifest, indent=2) + "\n")
    assert load(mp) == manifest, "Release manifest differs; inspect changes before updating it."
    print(json.dumps({"review_records": 80, "core": 9, "mixed": 4,
                      "utilization_views": 129, "continuations": 832,
                      "image_references": 276, "exact_pngs_verified": len(images),
                      "summary_recomputed": True, "release_hashes_verified": len(files),
                      "visual_truth_revalidated": False}, indent=2))


if __name__ == "__main__":
    main()
