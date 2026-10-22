"""Work package A (zero GPU): facts the next credit design must rest on. Writes audit_a.json.
1 versions: sha256 of the SFT-A config/tokenizer files, the plugin files in use, EasyR1 commit + local diff.
2 four-cell check: tool_credit.turn_value over every (case, sign of A); permuting the AFTER label never changes the
  output when the reader labels are valid (exhaustive), i.e. the rule is a two-cell before-gate.
3 dV centering: centring dV itself within a group of one question is identically 0 (dV is a per-question table value);
  centring s*dV (what v51_dvb implements) is 0 only when the whole group took the same first action.
4 real token spans: for 5 SFT-A crop-branch trajectories, tokenise each assistant turn with the SFT-A tokenizer and
  report token ranges of the WORD header ('<think>\\nStep k | WORD'), the caption, the action payload
  (<grounding>{...}</grounding> or <answer>X</answer>) and the end token, per turn; the issuing turn's caption precedes
  its own receipt (it can only use earlier views).
Usage: python audit_a.py <out_dir>"""
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
P = os.environ["PROJECT_DIR"]  # [scrub]
sys.path.insert(0, f"{P}/grpo_v51_credit")
import tool_credit as TC  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


M = f"{P}/outputs/sft_v22_turnwise_merged"
res = {"versions": {
    "sft_a": {f: sha(f"{M}/{f}") for f in ("config.json", "tokenizer.json", "preprocessor_config.json") if os.path.exists(f"{M}/{f}")},
    "plugins": {f"{d}/{f}": sha(f"{P}/{d}/{f}") for d in ("grpo_v51", "grpo_v51_credit", "grpo_v51_db")
                for f in ("caption_env.py", "easyr1_caption.py", "caption_contracts.py") if os.path.exists(f"{P}/{d}/{f}")},
    "easyr1_commit": subprocess.run(["git", "-C", os.environ["EASYR1_DIR"], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "easyr1_local_diff_sha": hashlib.sha256(subprocess.run(["git", "-C", os.environ["EASYR1_DIR"], "diff"], capture_output=True).stdout).hexdigest()[:16]}}

# 2 four-cell check
rows, invariant = [], True
for before, after, gold in itertools.product("AB", "AB", "A"):
    case = TC.case_of(before, after, gold)
    for a in (0.7, -0.7):
        flipped = TC.case_of(before, "B" if after == "A" else "A", gold)
        v, vf = TC.turn_value(case, a), TC.turn_value(flipped, a)
        invariant &= (v == vf)
        rows.append(dict(case=case, A=a, value=v, value_if_after_flipped=vf))
res["four_cell_check"] = dict(after_label_never_changes_output=invariant, table=rows)

# 3 dV centering
dvq = 0.5
groups = [("grounding", "grounding", "answer"), ("grounding",) * 3]
sgn = {"grounding": 1, "answer": -1}
res["dv_centering"] = {
    "centre_dV_itself": [round(dvq - dvq, 6) for _ in groups],
    "centre_s_dV": [[round(sgn[k] * dvq - sum(sgn[x] * dvq for x in g) / len(g), 4) for k in g] for g in groups]}

# 4 token spans
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(M)
spans = []
src = f"{P}/evidence/twobranch_20261019/traj_A_sft_crop.jsonl"
for line in list(open(src))[:5]:
    t = json.loads(line)
    for k, turn in enumerate(t["turns"], start=1):
        txt = turn["text"]
        m_head = re.match(r"(<think>\s*Step\s*\d+\s*\|\s*\w+)", txt)
        m_act = re.search(r"(<grounding>.*?</grounding>|<answer>.*?</answer>)\s*$", txt, re.S)
        ids = tok.encode(txt, add_special_tokens=False)

        def ntok(s):
            return len(tok.encode(s, add_special_tokens=False))
        head = ntok(m_head.group(1)) if m_head else 0
        act = ntok(m_act.group(1)) if m_act else 0
        spans.append(dict(uid=t["uid"], sample=t["sample_idx"], turn=k, kind=(turn.get("parsed") or {}).get("kind"),
                          n_tokens_text=len(ids), n_tokens_logged=turn.get("n_tokens"), word_header=[0, head],
                          caption=[head, len(ids) - act], action=[len(ids) - act, len(ids)],
                          end_token="<|im_end|> sampled by the policy (logged n_tokens includes it)" if turn.get("finish") == "stop" else turn.get("finish")))
res["token_spans"] = dict(note="ranges are token offsets within the turn's sampled text; receipts/prompt are outside "
                               "the response mask; the caption of a crop-issuing turn is written BEFORE its own receipt "
                               "exists, so it can only use earlier views", spans=spans)
res["unestablished_claims"] = [
    "E[A*N] sign = update direction of crop probability (docs/29, docs/30): correlational only",
    "E[c^2] = gradient variance (DB gate): second moment of a coefficient, not of c*score",
    "RR = redundant, WW = necessary-but-failed (frozen-reader cases): labels, not verified causes",
    "prefill letter agreement .777 = prefill bias: needs a natural-vs-natural agreement reference",
    "DB gate failure = method failure: the method was never trained"]
res["kept_records"] = "DB gate failure (t-db-wave1) kept as recorded; thresholds unchanged"
json.dump(res, open(out / "audit_a.json", "w"), indent=1)
print(json.dumps({k: (v if k != "token_spans" else f"{len(v['spans'])} turns") for k, v in res.items() if k != "four_cell_check"} |
                 {"four_cell_invariant": invariant}, indent=1)[:2500])
