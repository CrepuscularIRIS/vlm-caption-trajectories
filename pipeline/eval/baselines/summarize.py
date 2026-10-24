"""One scoring rule for every eval runner (eval_base direct/native, eval_v4 caption) -> trade-off table.

Per trajectory (mean over all question x sample rows, failures included):
  acc        last <answer>X</answer> in the final text == gold (fallback \\boxed{X} / "answer is X");
             truncated or errored trajectories count as wrong;
  fmt        the final text has an <answer>X</answer> tag;
  first      1 if at least one crop was executed;  later = max(crops - 1, 0);
  tokens     prompt_tokens + gen_tokens (every turn's full input incl. repeated history and image tokens, + output);
  truncated  generation hit the length / context budget.
usage: python summarize.py traj_A.jsonl [traj_B.jsonl ...] > table.json
"""
import json
import re
import sys
from collections import defaultdict

ANS_RE = re.compile(r"<answer>\s*\(?\s*([A-Z])\s*\)?\s*</answer>", re.I)
FALLBACK = (re.compile(r"\\boxed\{\s*\(?([A-Z])\)?\s*\}"), re.compile(r"answer\s*(?:is|:)\s*\**\s*\(?([A-Z])\)?\b", re.I))


def extract(text: str) -> tuple[str | None, int]:
    m = ANS_RE.findall(text or "")
    if m:
        return m[-1].upper(), 1
    for rx in FALLBACK:
        m = rx.findall(text or "")
        if m:
            return m[-1].upper(), 0
    return None, 0


def truncated(r: dict) -> bool:
    if r.get("end") in ("truncated", "overlong") or r.get("overlong"):
        return True
    return bool(r.get("turns")) and r["turns"][-1].get("finish") == "length"


def summarize(path: str) -> dict:
    tot, by_src = defaultdict(float), defaultdict(lambda: [0, 0])
    for line in open(path):
        r = json.loads(line)
        pred, fmt = extract(r.get("final_text") or "")
        ok = int(pred is not None and pred == r["gt"] and not r.get("error") and not truncated(r))
        c = int(r.get("n_crops", 0))
        for k, v in (("n", 1), ("acc", ok), ("fmt", fmt), ("first", c > 0), ("later", max(c - 1, 0)),
                     ("tokens", r["prompt_tokens"] + r["gen_tokens"]), ("gen", r["gen_tokens"]),
                     ("truncated", truncated(r)), ("error", bool(r.get("error")))):
            tot[k] += v
        by_src[r["source"]][0] += ok
        by_src[r["source"]][1] += 1
    n = tot.pop("n")
    out = {"file": path.rsplit("/", 1)[-1], "n": int(n), **{k: round(v / n, 4) for k, v in tot.items()}}
    out["by_source"] = {s: round(a / b, 4) for s, (a, b) in sorted(by_src.items())}
    return out


if __name__ == "__main__":
    print(json.dumps([summarize(p) for p in sys.argv[1:]], indent=1))
