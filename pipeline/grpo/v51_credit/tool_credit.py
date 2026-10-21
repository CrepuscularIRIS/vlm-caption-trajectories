"""Four-case tool credit (mentor 10-21): which tokens receive the outcome signal.

For every executed crop k of a training rollout, a frozen pixel-only reader R (Qwen3-VL-4B-Instruct served by vLLM,
same prompt as the offline gate frozen_reader.py: question with options, the views available, assistant turn
prefilled '<answer>(') reads the views before crop k (initial view + crops 1..k-1) and after it (+ crop k). Its answer
is compared with the gold letter, giving one case per crop:
  WR  wrong -> right   effective tool branch            turn tokens get max(A, 0)  (credit kept, never punished)
  RR  right -> right   redundant tool branch            turn tokens get min(A, 0)  (no share of the correct answer)
  WW  wrong -> wrong   necessary but failed exploration turn tokens get max(A, 0)  (not uniformly punished)
  RW  right -> wrong   misleading tool                  turn tokens get min(A, 0)  (blame kept, no credit)
'Turn tokens' = the assistant turn that issued crop k (the k-th contiguous run of the response mask: its caption,
<think> text and <grounding> action). The final answer turn and every rollout without crops keep the estimator's
advantage A unchanged. A crop whose reads are missing (no letter mass, request failure) or whose question has no gold
letter keeps A. CAPTION_CREDIT_RULE=none applies no change (identity, used for the parity smoke).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LETTERS = "ABCDEFGHIJ"
OPT = re.compile(r"^\(([A-Z])\)", re.M)
CASES = ("WR", "RR", "WW", "RW")


def rule() -> str:
    r = os.environ.get("CAPTION_CREDIT_RULE", "mentor4")
    if r not in ("mentor4", "none"):
        raise ValueError(f"CAPTION_CREDIT_RULE={r!r} not in ('mentor4', 'none')")
    return r


def gold_letter(gold) -> str | None:
    if isinstance(gold, dict):
        gold = gold.get("ground_truth", "")
    gold = str(gold).strip().upper()
    return gold if len(gold) == 1 and gold in LETTERS else None


def question_text(extra_info) -> str | None:
    if isinstance(extra_info, str):
        import ast
        try:
            extra_info = ast.literal_eval(extra_info)
        except (ValueError, SyntaxError):
            return None
    if isinstance(extra_info, dict) and isinstance(extra_info.get("question"), str):
        return extra_info["question"]
    return None


def case_of(before: str | None, after: str | None, gold: str | None) -> str | None:
    if gold is None or before is None or after is None:
        return None
    return ("R" if before == gold else "W") + ("R" if after == gold else "W")


def turn_value(case: str | None, a: float) -> float:
    if case in ("WR", "WW"):
        return max(a, 0.0)
    if case in ("RR", "RW"):
        return min(a, 0.0)
    return a


def segments(mask_row) -> list:
    """[start, end) of every contiguous run of 1s (one per assistant turn)."""
    out, s, n = [], None, len(mask_row)
    for t in range(n):
        if mask_row[t] and s is None:
            s = t
        elif not mask_row[t] and s is not None:
            out.append((s, t))
            s = None
    if s is not None:
        out.append((s, n))
    return out


def apply_credit(adv, mask, cases_list):
    """adv, mask: (B, L) tensors; cases_list[i] = case per executed crop of rollout i (crop k -> turn k)."""
    out = adv.clone()
    rows = mask.detach().cpu().tolist()
    changed = 0
    for i, cases in enumerate(cases_list):
        if not cases:
            continue
        segs = segments(rows[i])
        for k, case in enumerate(cases):
            if k >= len(segs) or case is None:
                continue
            s, e = segs[k]
            a = float(adv[i, s])
            v = turn_value(case, a)
            if v != a:
                out[i, s:e] = v
                changed += e - s
    return out, changed


# ---------------------------------------------------------------- reader client
def _png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def reader_messages(qtext: str, images) -> list:
    content = []
    for j, img in enumerate(images):
        content += [{"type": "text", "text": "Original image:" if j == 0 else f"Crop {j}:"},
                    {"type": "image_url", "image_url": {"url": _png(img)}}]
    content.append({"type": "text", "text": f"\n{qtext}\nAnswer with the option letter only."})
    return [{"role": "user", "content": content}, {"role": "assistant", "content": "<answer>("}]


def parse_top(top_logprobs, letters) -> tuple:
    """top_logprobs: [{'token': str, 'logprob': float}, ...] -> (argmax letter, confidence) over the options."""
    import math
    ps = {}
    for t in top_logprobs:
        tok = t["token"].strip()
        if tok in letters:
            ps[tok] = ps.get(tok, 0.0) + math.exp(t["logprob"])
    z = sum(ps.values())
    if z <= 0:
        return None, 0.0
    best = max(ps, key=ps.get)
    return best, ps[best] / z


def read_one(qtext: str, images, url: str, model: str, timeout: float = 120.0) -> tuple:
    letters = OPT.findall(qtext) or list(LETTERS)
    body = {"model": model, "messages": reader_messages(qtext, images), "max_tokens": 1, "temperature": 0.0,
            "logprobs": True, "top_logprobs": 20, "continue_final_message": True, "add_generation_prompt": False}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read())
    top = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return parse_top(top, letters)


def annotate(states, mms, workers: int = 64) -> dict:
    """Set state['cases'] for every rollout: reads views 0..n_crops once each, then one case per crop."""
    url, model = os.environ["CAPTION_READER_URL"], os.environ["CAPTION_READER_MODEL"]
    jobs = []
    for i, (st, mm) in enumerate(zip(states, mms)):
        st["cases"] = []
        n = st["env"].crops
        info = mm["env"]
        if n == 0 or not info.get("qtext") or not info.get("gold"):
            st["cases"] = [None] * n
            continue
        for j in range(n + 1):
            jobs.append((i, j))
    reads, fails = {}, 0

    def work(job):
        i, j = job
        st, mm = states[i], mms[i]
        try:
            return job, read_one(mm["env"]["qtext"], st["images"][: j + 1], url, model)
        except Exception as e:  # recorded per crop as a missing read (A unchanged), counted
            return job, ("ERR", repr(e)[:200])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for job, res in ex.map(work, jobs):
            if res[0] == "ERR":
                fails += 1
                reads[job] = (None, 0.0)
            else:
                reads[job] = res
    counts = {c: 0 for c in CASES}
    counts["none"] = 0
    for i, (st, mm) in enumerate(zip(states, mms)):
        n = st["env"].crops
        if not n or st["cases"]:
            continue
        gold = mm["env"]["gold"]
        st["cases"] = [case_of(reads[(i, k - 1)][0], reads[(i, k)][0], gold) for k in range(1, n + 1)]
        st["reads"] = [list(reads[(i, j)]) for j in range(n + 1)]
        for c in st["cases"]:
            counts[c if c else "none"] += 1
    print(f"[tool_credit] reads={len(jobs)} failed={fails} cases={counts}", flush=True)
    return counts


def install_advantage_patch():
    import verl.trainer.ray_trainer as rt
    if getattr(rt.compute_advantage, "_credit_patched", False):
        return
    original = rt.compute_advantage

    def compute_advantage(data, *args, **kwargs):
        data = original(data, *args, **kwargs)
        mm = data.non_tensor_batch.get("multi_modal_data")
        if mm is None or not len(mm) or "caption_cases" not in mm[0]:
            return data
        r = rule()
        cases = [list(m.get("caption_cases") or []) for m in mm]
        old = data.batch["advantages"]
        if r == "none":
            new, changed = old, 0
        else:
            new, changed = apply_credit(old, data.batch["response_mask"], cases)
            new = new.to(old.dtype)
        data.batch["advantages"] = new
        data.batch["returns"] = data.batch["returns"] + (new - old)
        m = data.batch["response_mask"].bool()
        pos_old = float(old[m].clamp(min=0).sum()); neg_old = float(old[m].clamp(max=0).sum())
        pos_new = float(new[m].clamp(min=0).sum()); neg_new = float(new[m].clamp(max=0).sum())
        print(f"[tool_credit] rule={r} tokens_changed={changed} pos_mass {pos_old:.1f}->{pos_new:.1f} "
              f"neg_mass {neg_old:.1f}->{neg_new:.1f}", flush=True)
        return data

    compute_advantage._credit_patched = True
    rt.compute_advantage = compute_advantage
