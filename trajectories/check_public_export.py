#!/usr/bin/env python3
"""Validate the public trajectory export (library A and B) before publishing.

Always runs (no private data needed):
  * record and step totals, per-record consistency (steps, crops, supervision, induced misses);
  * forbidden keys at any depth (image/images/image_path/question/options/prompt/path/...);
  * forbidden strings: machine paths, image URLs, base64 image blobs, API keys, endpoints, IPs;
  * file size limit (50 MB per file); system_prompts.json is scanned with the same string rules;
  * STATS.json agrees with numbers recomputed from the export.

With --pool (and optionally --source-a / --source-b), also runs an exact-substring scan of every
source question text, question stem, displayed prompt and option text (>= 20 chars) against every
string in the export. Any hit outside an assistant turn's ``text`` fails. Hits inside assistant
text are the generator's own words; they are reported (assistant turns per kind) and must equal
STATS.json -> library_*.question_quotes.

Exit status 0 = clean, 1 = violation(s). Usage:
  python check_public_export.py [--pool pool.jsonl --source-a caption_sft_v2_2_keep.json --source-b DATABASE.jsonl]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {"A": "library_A_v2_2.jsonl", "B": "library_B_v4c.jsonl"}
EXPECT = {
    "A": {"records": 4040, "assistant_steps": 7047, "supervised_assistant_steps": 7047,
          "in_sft_aligned_3983": 3983, "assistant_steps_in_sft_aligned_3983": 6953},
    "B": {"records": 3983, "assistant_steps": 14826, "supervised_assistant_steps": 9898,
          "induced_misses_total": 4928},
}
MAX_BYTES = 50 * 1024 * 1024
WIN = 20

FORBIDDEN_KEYS = {"image", "images", "image_path", "image_paths", "image_url", "image_urls", "img",
                  "question", "questions", "question_text", "options", "option", "choices",
                  "prompt", "prompts", "conversations", "messages", "result_path", "source_result",
                  "path", "paths", "url", "urls", "file", "filename", "bytes", "base64", "pixels"}
FORBIDDEN_KEY_PARTS = ("path", "base64", "b64", "url")

# Fail anywhere (including assistant text).
HARD_PATTERNS = {
    "machine_path": re.compile(r"/home/|/root/|/data/|/tmp/|/mnt/|/scratch/|~/|lingxufeng|icml-envs|bench_downloads"),
    "image_url": re.compile(r"https?://\S+\.(?:png|jpe?g|webp|gif|bmp|tiff?)\b", re.I),
    "data_uri": re.compile(r"data:image/|;base64,", re.I),
    "base64_blob": re.compile(r"[A-Za-z0-9+/]{160,}={0,2}"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}|\bghp_[A-Za-z0-9]{20,}|github_pat_|\bAKIA[0-9A-Z]{16}\b|\bxox[bap]-|Bearer\s+[A-Za-z0-9._\-]{16,}"),
    "endpoint": re.compile(r"api\.openai\.com|api\.x\.ai|api\.anthropic\.com|generativelanguage\.googleapis|"
                           r"cliproxy|localhost|127\.0\.0\.1|/v1/chat/completions|/v1/messages|OPENAI_API|API_KEY", re.I),
}
# Fail outside assistant text / review summary; report inside (models transcribe URLs or numbers seen in the image).
SOFT_PATTERNS = {
    "any_url": re.compile(r"https?://", re.I),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"),
}


def walk(obj, where: str, in_assistant_text: bool = False):
    """Yield (key_path, key, value, is_assistant_text) for every key and string leaf."""
    if isinstance(obj, dict):
        is_asst = obj.get("role") == "assistant"
        for k, v in obj.items():
            yield where, k, None, False
            yield from walk(v, f"{where}.{k}", is_asst and k == "text")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{where}[{i}]", in_assistant_text)
    elif isinstance(obj, str):
        yield where, None, obj, in_assistant_text


def load(lib: str, root: Path) -> list:
    path = root / FILES[lib]
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def recompute(lib: str, recs: list) -> dict:
    asst = [t for r in recs for t in r["turns"] if t["role"] == "assistant"]
    out = {"records": len(recs), "assistant_steps": len(asst),
           "supervised_assistant_steps": sum(bool(t.get("supervised")) for t in asst),
           "crops_total": sum(r["crops"] for r in recs),
           "direct_answer": sum(r["steps"] == 1 for r in recs),
           "per_benchmark": dict(Counter(r["bench"] for r in recs)),
           "generators": dict(Counter(r["generator_model"] for r in recs))}
    if lib == "A":
        out["in_sft_aligned_3983"] = sum(r["in_sft_aligned_3983"] for r in recs)
        out["assistant_steps_in_sft_aligned_3983"] = sum(r["steps"] for r in recs if r["in_sft_aligned_3983"])
    else:
        out["induced_misses_total"] = sum(r["induced_misses"] for r in recs)
    return out


def check_record(lib: str, r: dict, errors: list) -> None:
    tid = r.get("audit_id")
    asst = [t for t in r["turns"] if t["role"] == "assistant"]
    if len(asst) != r["steps"]:
        errors.append(f"{lib}:{tid} steps {r['steps']} != assistant turns {len(asst)}")
    if sum(t["action"] == "crop" for t in asst) != r["crops"]:
        errors.append(f"{lib}:{tid} crops mismatch")
    if asst and asst[-1]["action"] != "answer":
        errors.append(f"{lib}:{tid} last assistant turn is not an answer")
    first = r["turns"][0]
    if first["role"] != "user" or not first["text"].startswith("Question: [question and options removed"):
        errors.append(f"{lib}:{tid} first user turn does not carry the question placeholder")
    if r.get("gold_letter") is not None and not re.fullmatch(r"[A-Z0-9]", str(r["gold_letter"])):
        errors.append(f"{lib}:{tid} gold_letter is not a single option label")
    if lib == "B":
        sup = [t["supervised"] for t in asst]
        if sup != r["assistant_supervision"]:
            errors.append(f"{lib}:{tid} supervision mask mismatch")
        induced = sorted(t["step"] for t in asst if t.get("induced_miss"))
        if induced != sorted(r["induced_steps"]) or len(induced) != r["induced_misses"]:
            errors.append(f"{lib}:{tid} induced-miss markers mismatch")
        if any(t["supervised"] == bool(t.get("induced_miss")) for t in asst):
            errors.append(f"{lib}:{tid} induced turn supervised or clean turn unsupervised")


def scan_strings(lib: str, recs: list, errors: list, report: Counter) -> None:
    for r in recs:
        tid = r.get("audit_id")
        for where, key, val, in_asst in walk(r, f"{lib}:{tid}"):
            if key is not None:
                kl = key.lower()
                if kl in FORBIDDEN_KEYS or any(p in kl for p in FORBIDDEN_KEY_PARTS):
                    errors.append(f"forbidden key '{key}' at {where}")
                continue
            for name, pat in HARD_PATTERNS.items():
                m = pat.search(val)
                if m:
                    errors.append(f"{name} at {where}: {m.group(0)[:60]!r}")
            for name, pat in SOFT_PATTERNS.items():
                if pat.search(val):
                    if in_asst:
                        report[f"{lib}.assistant_text_with_{name}"] += 1
                    elif where.endswith(".review.summary_zh"):
                        report[f"{lib}.review_summary_with_{name}"] += 1   # reviewer transcribing image text
                    else:
                        errors.append(f"{name} outside assistant text at {where}")


def question_needles(pool_path: Path, src_a: Path | None, src_b: Path | None, ids: set) -> dict:
    """Return {needle_text: kind} for every question-derived string >= WIN chars."""
    needles = {}
    with pool_path.open(encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            if it["item_id"] not in ids:
                continue
            q = it["question"]
            needles.setdefault(q, "question")
            needles.setdefault(q.split("\n(A)")[0].strip(), "stem")
            for o in re.findall(r"^\([A-J]\)\s*(.+)$", q, re.M):
                needles.setdefault(o.strip(), "option")
    marker = "Original image [original_image]"
    convs = []
    if src_a:
        convs += [row["conversations"] for row in json.loads(src_a.read_text(encoding="utf-8"))]
    if src_b:
        convs += [json.loads(line)["conversations"] for line in src_b.open(encoding="utf-8")]
    for conv in convs:
        shown = conv[1]["value"].split(marker)[0]
        needles.setdefault(shown.strip(), "displayed_prompt")
        body = shown.removeprefix("Question: ").strip()
        needles.setdefault(body, "displayed_prompt")
        for line in body.split("\n"):
            needles.setdefault(line.strip(), "displayed_line")
    return {n: k for n, k in needles.items() if len(n) >= WIN}


def scan_questions(lib: str, recs: list, needles: dict, errors: list, report: Counter) -> None:
    index = defaultdict(list)
    for n in needles:
        index[n[:WIN]].append(n)
    for r in recs:
        tid = r.get("audit_id")
        for where, key, val, in_asst in walk(r, f"{lib}:{tid}"):
            if key is not None or len(val) < WIN:
                continue
            hits = set()
            for i in range(len(val) - WIN + 1):
                cands = index.get(val[i:i + WIN])
                if cands:
                    hits.update(n for n in cands if val.startswith(n, i))
            if in_asst:
                for kind in {needles[n] for n in hits}:
                    report[f"{lib}.assistant_turns_quoting_{kind}"] += 1
                continue
            for n in hits:
                errors.append(f"question-derived text ({needles[n]}) outside assistant text at {where}: {n[:50]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=HERE)
    ap.add_argument("--pool", type=Path, help="private item pool (pool.jsonl) with question texts")
    ap.add_argument("--source-a", type=Path, help="private caption_sft_v2_2_keep.json")
    ap.add_argument("--source-b", type=Path, help="private interp v4c DATABASE.jsonl")
    args = ap.parse_args()

    errors: list = []
    report: Counter = Counter()
    for p in sorted(args.root.iterdir()):
        if p.is_file() and p.stat().st_size > MAX_BYTES:
            errors.append(f"{p.name} is {p.stat().st_size / 2**20:.1f} MB > 50 MB")

    stats = json.loads((args.root / "STATS.json").read_text(encoding="utf-8"))
    libs = {lib: load(lib, args.root) for lib in FILES}
    for lib, recs in libs.items():
        got = recompute(lib, recs)
        print(f"[{lib}] " + ", ".join(f"{k}={v}" for k, v in got.items() if not isinstance(v, dict)))
        for k, v in EXPECT[lib].items():
            if got[k] != v:
                errors.append(f"{lib}: {k}={got[k]} expected {v}")
        skey = "library_A_v2_2" if lib == "A" else "library_B_v4c"
        for k in ("records", "assistant_steps", "supervised_assistant_steps", "crops_total", "direct_answer",
                  "per_benchmark", "generators"):
            if stats[skey][k] != got[k]:
                errors.append(f"{lib}: STATS.json {k} disagrees with export")
        ids = [r["item_id"] for r in recs]
        if len(set(ids)) != len(ids):
            errors.append(f"{lib}: duplicate item_id")
        for r in recs:
            check_record(lib, r, errors)
        scan_strings(lib, recs, errors, report)

    prompts = json.loads((args.root / "system_prompts.json").read_text(encoding="utf-8"))
    for pid, entry in prompts.items():
        for name, pat in HARD_PATTERNS.items():
            if pat.search(entry["text"]):
                errors.append(f"{name} in system_prompts.json:{pid}")
    used = {r["system_prompt_id"] for recs in libs.values() for r in recs}
    if used != set(prompts):
        errors.append("system_prompt_id values do not match system_prompts.json")

    aligned = {r["item_id"] for r in libs["A"] if r["in_sft_aligned_3983"]}
    if aligned != {r["item_id"] for r in libs["B"]}:
        errors.append("A.in_sft_aligned_3983 set differs from the B item set")

    if args.pool:
        all_ids = {r["item_id"] for recs in libs.values() for r in recs}
        needles = question_needles(args.pool, args.source_a, args.source_b, all_ids)
        print(f"[questions] scanning {len(needles)} question-derived strings (>= {WIN} chars)")
        for lib, recs in libs.items():
            scan_questions(lib, recs, needles, errors, report)
        for lib, skey in (("A", "library_A_v2_2"), ("B", "library_B_v4c")):
            exp = stats[skey].get("question_quotes", {})
            for kind in ("question", "stem", "displayed_prompt", "displayed_line", "option"):
                got = report[f"{lib}.assistant_turns_quoting_{kind}"]
                if got != exp.get(kind, 0):
                    errors.append(f"{lib}: assistant turns quoting {kind} = {got}, STATS.json says {exp.get(kind, 0)}")
    else:
        print("[questions] skipped: pass --pool (and --source-a/--source-b) to run the exact-substring scan")

    print("[report] " + (", ".join(f"{k}={v}" for k, v in sorted(report.items())) or "none"))
    if errors:
        print(f"FAIL: {len(errors)} violation(s)")
        for e in errors[:50]:
            print("  " + e)
        return 1
    print("PASS: no violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
