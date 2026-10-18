#!/usr/bin/env python3
"""MathVerse 及自由作答的排版无关评分器(三席审查修复,2026-09-21)。

设计要点(对应审查 H2/M1/L3/Sol-高):
- 候选抽取扫**全文**所有位置(bold / "answer is" / \\boxed / 首行 / 末行),不再只看尾部;
- 严格优先:选项字母(动态 A–F)、数值序列末项等价(坐标=全序列等价)、归一化全串相等;
- 数值 token 兜底才允许包含,禁止任意子串(杜绝 'cm' vs '254.47 cm²' 假阳);
- 仍判 False 的 free-form 行走 LLM judge(claude-opus-5,与 MathVerse 官方 GPT-judge 同范式);
- p6 式散文捞字母:要求命中处近邻有 answer/option 线索,否则不采信。

judge: judge_freeform(pred, gold, full_raw) -> (bool, reason)
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_NUM = r"-?\d+(?:\.\d+)?"

_CANDIDATE_PATTERNS = [
    re.compile(r"answer\s+(?:is|:)\s*:?\s*\${0,2}([^$\n]{1,80})", re.I),
    re.compile(r"\*{1,3}([^*\n]{1,60})\*{1,3}"),
    re.compile(r"\\boxed\{([^{}]{1,60})\}"),
    re.compile(r"=\s*\$?(-?[^\s$,.;]{1,30})\$?\s*(?:[.,;]|$)", re.M),
]


def strip_latex(s: str) -> str:
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\circ", "°").replace("\\times", "×").replace("\\cdot", "·")
    s = s.replace("\\pi", "π").replace("\\infty", "∞").replace("\\sqrt", "√").replace("\\frac", "")
    s = re.sub(r"[${}]", "", s)
    return s.strip()


def norm(s: str) -> str:
    s = strip_latex(s or "")
    s = re.sub(r"[\s,_\\^]", "", s)
    s = re.sub(r"(?<=\d)\.0+(?=[^\d.]|$)", "", s)
    return s.lower().rstrip(".;")


def num_tokens(s: str) -> list[float]:
    return [float(m) for m in re.findall(_NUM, strip_latex(s))]


def nums_eq(a: list[float], b: list[float]) -> bool:
    """数值序列等价:坐标/多值需全序列相等;单值比末项。容差:绝对 1e-6(舍入级差异不算对)。"""
    def close(x: float, y: float) -> bool:
        return abs(x - y) <= max(1e-6, 1e-9 * max(abs(x), abs(y)))
    if not a or not b:
        return False
    if len(a) == len(b):
        return all(close(x, y) for x, y in zip(a, b))
    return close(a[-1], b[-1])


def _numonly(s: str) -> bool:
    """剥掉前导标签残留(=:/)与尾部单位(字母/°%/~)后是否纯数值。含变量/根式/π 一律 False。"""
    s2 = re.sub(r"[a-z°%~]+$", "", s)
    s2 = re.sub(r"^[=:\s]+", "", s2)
    return bool(re.fullmatch(r"[-+±()°%.,\d\s]+", s2))


def _strip_label(s: str) -> str:
    """剥掉前导英文标签('Circumference =' / 'Answer:');不碰反斜杠(那是 LaTeX 命令)。"""
    return re.sub(r"^[A-Za-z .]{1,30}=?", "", s or "", count=1)


def candidate_strings(content: str) -> list[str]:
    """全文候选 span,首行/末行兜底;不依赖答案出现位置(排版无关)。"""
    out: list[str] = []
    for pat in _CANDIDATE_PATTERNS:
        out += [m.group(1) for m in pat.finditer(content or "")]
    lines = [l.strip() for l in (content or "").splitlines() if l.strip()]
    if lines:
        out.append(lines[0][:80])
        out.append(lines[-1][:80])
    return out


def valid_letters(question: str) -> str:
    """逐题合法字母集:行式 'A: x' 或内联 '(a) x'。"""
    ls = re.findall(r"^([A-F])[:.)]", question or "", re.M)
    if not ls:
        ls = re.findall(r"\(([A-Fa-f])\)", question or "")
    ls = [x.upper() for x in ls]
    return "".join(sorted(set(ls))) or "ABCD"


def extract_option_letter(content: str, letters: str) -> tuple[str | None, str]:
    """动态字母集 + 线索约束抽取。"""
    c = content or ""
    for pat in (r"(?:answer|option|choice)\s+is\s*:?\s*\*{0,2}\(?([A-F])\)?\b",
                r"answer\s*:?\s*\*{0,2}\(?([A-F])\)?\b"):
        ms = [m for m in re.finditer(pat, c, re.I) if m.group(1).upper() in letters]
        if ms:
            return ms[-1].group(1).upper(), "explicit"
    ms = [m for m in re.finditer(r"\(([A-Fa-f])\)", c) if m.group(1).upper() in letters]
    if ms:
        return ms[-1].group(1).upper(), "paren"
    lines = [l.strip() for l in c.splitlines() if l.strip()]
    for which in ("first_line", "last_line"):
        if not lines:
            break
        line = lines[0] if which == "first_line" else lines[-1]
        m = re.fullmatch(r"\*{0,2}\(?([A-Fa-f])\)?\*{0,2}[,.]?", line)
        if m and m.group(1).upper() in letters:
            return m.group(1).upper(), which
    return None, "none"


def letter_by_option_text(content: str, question: str) -> str | None:
    """答案给的是选项文本而非字母('rubber'→A):内联/行式选项文本反查。"""
    opts = re.findall(r"\(([A-Fa-f])\)\s*([^()\n]{4,40})", question or "")
    opts += [(m.group(1), m.group(2).strip()) for m in
             re.finditer(r"^([A-Fa-f])[:.)]\s*(.{4,60})$", question or "", re.M)]
    tail = (content or "")[-400:]
    hits = [(L, t) for L, t in opts if t and t in tail]
    if len(hits) == 1:
        return hits[0][0].upper()
    return None


def score_freeform_strict(content: str, gold: str) -> tuple[bool, str]:
    """排版无关的严格评分;成功路径不打 API。数值路径要求两侧均可纯数值化(带尾单位可)。"""
    gnorm = norm(gold)
    gnums = num_tokens(gold)
    g_numonly = _numonly(norm(gold))
    cands = candidate_strings(content)
    # 1) 归一化全串相等
    for c in cands:
        if gnorm and norm(c) == gnorm:
            return True, "norm_eq"
    # 2) 数值序列等价(坐标=全序列;单值=末项);gold/candidate 先剥前导标签
    if g_numonly or _numonly(norm(_strip_label(gold))):
        for c in cands:
            cn = norm(_strip_label(c))
            if _numonly(cn):
                cnums = num_tokens(_strip_label(c))
                if cnums and gnums and nums_eq(cnums, gnums):
                    return True, "nums_eq"
    return False, "no_strict"


def _judge_call(prompt: str, max_tokens: int = 2048) -> str:
    base = os.environ["LITELLM_BASE_URL"].rstrip("/")
    key = os.environ["LITELLM_API_KEY"]
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({"model": os.environ.get("JUDGE_MODEL", "claude-opus-5"),
                         "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"]


def judge_freeform(pred_content: str, gold: str) -> tuple[bool | None, str]:
    """LLM judge 兜底(仅严格评分失败时调用)。失败返回 (None, 'judge_error')——
    绝不把通道故障降级为负判(2026-09-21 审查追加要求)。"""
    prompt = (
        "Judge whether the model's final answer is mathematically equivalent to the gold answer. "
        f"Gold: {gold!r}\nModel output (may contain reasoning):\n---\n{pred_content[-2000:]}\n---\n"
        "Reply with exactly one line: EQUIVALENT or NOT, then a short reason after a pipe |.")
    try:
        out = _judge_call(prompt)
        first = out.strip().splitlines()[0].upper()
        return first.startswith("EQUIVALENT"), f"judge:{'eq' if first.startswith('EQUIVALENT') else 'ne'}"
    except Exception as e:  # noqa: BLE001 — 通道故障 → 未判定,不是 False
        logger.warning("judge 失败: %s", e)
        return None, "judge_error"


def score_mathverse(content: str, question: str, gold: str, qtype: str,
                    use_judge: bool = True) -> dict[str, Any]:
    """MathVerse 单行评分入口。gold:选择题=字母;free-form=值。"""
    if qtype == "multi-choice":
        letters = valid_letters(question)
        letter, method = extract_option_letter(content, letters)
        if letter is None:
            # 值→字母兜底:回答尾值与选项数值匹配
            gnums_cands = [num_tokens(c) for c in candidate_strings(content)]
            for L, text in re.findall(r"^([A-F])[:.)]\s*(.+)$", question or "", re.M):
                onums = num_tokens(text)
                if onums and any(cn and cn[-1] == onums[-1] for cn in gnums_cands if cn):
                    letter, method = L, "value_to_letter"
                    break
        ok = letter is not None and letter.upper() == str(gold).strip().upper()
        return {"answer": letter, "correct": bool(ok), "method": method}
    strict, method = score_freeform_strict(content, gold)
    if strict:
        return {"answer": gold, "correct": True, "method": method}
    if use_judge:
        ok, m = judge_freeform(content, gold)
        if ok:
            return {"answer": gold, "correct": True, "method": m}
    return {"answer": None, "correct": False, "method": "fail"}
