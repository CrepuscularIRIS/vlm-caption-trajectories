"""Export E1 trajectories (552 questions x 4 samples per condition) as public rollout files.

One gzipped JSONL per condition in trajectories/e1_rollouts/<id>.jsonl.gz, one line per rollout, assistant turns
verbatim. Question text, options, images and machine paths are not in the source trajectories and are not added;
every string is scanned with the HARD patterns of trajectories/check_public_export.py (any hit fails the export).
Conditions and display names come from logs/baselines_v52_20261024/summary.json.

Assistant turns that restate the question stem (first line of the question, >= 40 chars, whitespace-insensitive) get
it replaced by "[question stem removed]"; counts are in STATS.json. The question text is read from a local review file
(fields uid, question) that is not published.

usage: python export_e1_rollouts.py TRAJ_DIR QUESTIONS_JSONL
       (TRAJ_DIR holds traj_<tag>.jsonl; QUESTIONS_JSONL = any review_<tag>.jsonl from pool/build_review.py)
"""
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "trajectories"), str(Path(__file__).resolve().parent)]
from check_public_export import HARD_PATTERNS  # noqa: E402
from summarize import extract, truncated  # noqa: E402  (same scoring as summary.json)

OUT = REPO / "trajectories" / "e1_rollouts"
TURN_KEYS = ("finish", "n_tokens", "check_errors", "tools", "tool", "box_abs")
# Model-written fake tool output (DeepEyes writes made-up image URLs / base64 after </tool_call>) is replaced by a
# marker and counted in STATS.json; machine paths and API keys still fail the export.
REDACT = ("image_url", "data_uri", "base64_blob", "endpoint")
REDACTED = Counter()


def redact(text: str) -> str:
    for k in REDACT:
        text, n = HARD_PATTERNS[k].subn(f"[removed:{k}]", text)
        REDACTED[k] += n
    return text


STEMS: dict[str, re.Pattern] = {}


def load_stems(path: str) -> None:
    for line in open(path):
        r = json.loads(line)
        stem = r["question"].split("\n")[0].strip()
        if len(stem) >= 40 and r["uid"] not in STEMS:
            STEMS[r["uid"]] = re.compile(r"\s+".join(map(re.escape, stem.split())))


def unquote(uid: str, text: str) -> str:
    if uid not in STEMS:
        return text
    text, n = STEMS[uid].subn("[question stem removed]", text)
    REDACTED["question_stem"] += n
    return text


def turn(t: dict, uid: str) -> dict:
    out = {"role": "assistant", "step": t["round"] + 1, "text": unquote(uid, redact(t["text"]))}
    if isinstance(t.get("parsed"), dict):
        out["kind"] = t["parsed"].get("kind")
    out.update({k: json.loads(redact(json.dumps(t[k]))) for k in TURN_KEYS if k in t})
    return out


def rollout(cond: str, r: dict) -> dict:
    turns = [turn(t, r["uid"]) for t in r["turns"]]
    pred, fmt = extract(r.get("final_text") or "")
    ok = int(pred is not None and pred == r["gt"] and not r.get("error") and not truncated(r))
    return {"condition": cond, "bench": r["source"], "item_id": r["uid"], "sample_idx": r["sample_idx"],
            "gold_letter": r["gt"], "final_answer": pred, "has_answer_tag": fmt, "correct": ok,
            "correct_train_rule": r.get("ok_train_rule"), "steps": len(turns), "crops": r["n_crops"],
            "boxes_abs": r.get("boxes_abs"), "image_wh": r.get("image_wh"), "end": r.get("end"), "truncated": truncated(r),
            "error": r.get("error"), "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["gen_tokens"],
            "turns": turns}


def strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)
    elif isinstance(obj, str):
        yield obj


def main() -> None:
    src = Path(sys.argv[1])
    load_stems(sys.argv[2])
    assert len(STEMS) > 400, len(STEMS)
    OUT.mkdir(parents=True, exist_ok=True)
    stats, bad = {}, []
    for c in json.load(open(REPO / "logs/baselines_v52_20261024/summary.json")):
        rows = [rollout(c["id"], json.loads(x)) for x in open(src / c["file"])]
        for r in rows:
            if r["item_id"] in STEMS and any(STEMS[r["item_id"]].search(s) for s in strings(r)):
                bad.append((c["id"], r["item_id"], "question_stem"))
            for s in strings(r):
                bad += [(c["id"], r["item_id"], k) for k, rx in HARD_PATTERNS.items() if rx.search(s)]
        acc = round(sum(r["correct"] for r in rows) / len(rows), 4)
        assert len(rows) == c["n"] == 2208 and acc == c["acc"], (c["id"], len(rows), acc, c["acc"])
        with gzip.open(OUT / f"{c['id']}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[c["id"]] = {"name": c["name"], "label": c["label"], "rollouts": len(rows), "acc": acc,
                          "steps": dict(Counter(r["steps"] for r in rows)), "crops_total": sum(r["crops"] for r in rows),
                          "redacted_in_assistant_text": dict(REDACTED)}
        REDACTED.clear()
    if bad:
        sys.exit(f"forbidden strings: {Counter(k for *_, k in bad)}; first {bad[:5]}")
    json.dump(stats, open(OUT / "STATS.json", "w"), ensure_ascii=False, indent=1)
    print(len(stats), "conditions,", sum(s["rollouts"] for s in stats.values()), "rollouts; acc matches summary.json")


if __name__ == "__main__":
    main()
