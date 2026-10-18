#!/usr/bin/env python3
"""API t=0 baseline: glm-5.3-flash 直接作答,五个图文 benchmark 的准确率。

用途(Handoff §4):E3 的 t=0 对照、E1 的 TO/TD/VO 准确率上下文、prompt→抽取→评分链路验证。
VSI-Bench 是视频,本通道 video_url 被上游静默丢弃(2026-09-21 实测),留本地 GPU 轮。

Usage:
    python3 obs/src/run_api_t0.py --benchmarks mmvp --limit 8   # 试点
    python3 obs/src/run_api_t0.py                                # 全量(断点续跑)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from common import BENCH, REPO, git_commit, load_subset  # noqa: E402
from scoring import extract_option_letter, score_mathverse, valid_letters  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("api_t0")

MAX_PIXELS = 1_003_520  # 与本地 Qwen 协议一致(1280*28*28),超限等比缩
RPM = 55                # flash 条目限 60,留余量
WORKERS = 6
MAX_TOKENS = 4096
TEMPERATURE = 0.0
EXP = "api_t0"
MODEL_ENV = "LITELLM_SMOKE_MODEL"

LETTER_PATTERNS = [
    re.compile(r"answer\s+is\s*:?\s*\*{0,2}\(([A-Da-d])\)", re.I),
    re.compile(r"answer\s+is\s*:?\s*\*{0,2}([A-Da-d])\b", re.I),
    re.compile(r"[Aa]nswer[:\s]+\(([A-Da-d])\)"),
    re.compile(r"\(([A-Da-d])\)"),
    re.compile(r"^\*{0,2}([A-Da-d])\*{0,2}[,.]?\s*$", re.M),
    re.compile(r"\b([A-D])\b"),
]


class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm
        self._next_t = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_t)
            self._next_t = start + self._interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def load_env() -> tuple[str, str, str]:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    for k in ("LITELLM_BASE_URL", "LITELLM_API_KEY", MODEL_ENV):
        if not os.environ.get(k):
            raise KeyError(f".env 缺 {k}")
    return os.environ["LITELLM_BASE_URL"].rstrip("/"), os.environ["LITELLM_API_KEY"], os.environ[MODEL_ENV]


def image_b64(path: str) -> tuple[str, str]:
    """原生字节优先(≤8MB 不动,保真);超限才等比缩到 ≤8MB 再编码 JPEG。返回 (b64, mime)。"""
    from PIL import Image
    raw = Path(path).read_bytes()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(Path(path).suffix.lower(), "image/jpeg")
    if len(raw) <= 8 << 20:
        return base64.b64encode(raw).decode(), mime
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = ((8 << 20) / len(raw)) ** 0.5
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def call_flash(base: str, key: str, model: str, text: str, img_path: str | None,
               limiter: RateLimiter, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    content: Any = [{"type": "text", "text": text}]
    if img_path:
        b64, mime = image_b64(img_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    payload = json.dumps({
        "model": model, "temperature": TEMPERATURE, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }).encode()
    last_err = ""
    attempts = 0
    for attempt in range(6):
        attempts += 1
        limiter.wait()
        req = urllib.request.Request(
            base + "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.monotonic()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            msg = d["choices"][0]["message"]
            det = (d.get("usage", {}) or {}).get("completion_tokens_details", {}) or {}
            return {"content": msg.get("content") or "",
                    "reasoning_content": msg.get("reasoning_content") or "",
                    "finish_reason": d["choices"][0].get("finish_reason"),
                    "latency": round(time.monotonic() - t0, 2),
                    "tokens_out": d.get("usage", {}).get("completion_tokens"),
                    "reasoning_tokens": det.get("reasoning_tokens"), "attempts": attempts}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode()[:150]}"
            if e.code == 429:
                time.sleep(20)
            else:
                time.sleep(min(300, 5 * 2 ** attempt))
        except Exception as e:  # noqa: BLE001 — 网络/JSON 异常统一重试
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(min(300, 5 * 2 ** attempt))
    return {"content": "", "reasoning_content": "", "finish_reason": "error",
            "latency": None, "tokens_out": None, "reasoning_tokens": None,
            "attempts": attempts, "error": last_err}


def extract_letter(content: str) -> tuple[str | None, str]:
    for i, pat in enumerate(LETTER_PATTERNS):
        ms = list(pat.finditer(content))
        if ms:
            return ms[-1].group(1).upper(), f"p{i+1}"
    return None, "none"


# ---------- 每 benchmark:构建请求与评分 ----------

def build_requests(name: str) -> list[dict[str, Any]]:
    rows = load_subset(name)
    reqs = []
    if name == "mathverse":
        for r in rows:
            mc = r["question_type"] == "multi-choice"
            reqs.append({"item_id": r["item_id"], "condition": r["problem_version"],
                         "text": r["prompt_wo"], "image": r.get("image_abs"),
                         "gold": r["answer"].strip().upper() if mc else str(r["answer"]).strip(),
                         "qtype": r["question_type"],
                         "question": r.get("question", "")})
    elif name == "hr_bench":
        for r in rows:
            for v in r["variants"]:
                text = (f"{r['question']}\nChoices:\nA: {v['A']}\nB: {v['B']}\nC: {v['C']}\nD: {v['D']}"
                        "\nAnswer with the option's letter only.")
                reqs.append({"item_id": r["item_id"], "condition": f"cycle_{v['cycle_category']}",
                             "text": text, "image": r.get("image_abs"),
                             "gold": v["answer_letter"].strip().upper(),
                             "qtype": "multi-choice", "question": text})
    elif name == "mmstar":
        for r in rows:
            text = f"{r['question']}\nAnswer with the option's letter only."
            reqs.append({"item_id": r["item_id"], "condition": "t0",
                         "text": text, "image": r.get("image_abs"),
                         "gold": r["answer"].strip().upper(),
                         "qtype": "multi-choice", "question": r["question"]})
    elif name == "mmvp":
        for r in rows:
            text = f"{r['question']}\nAnswer with (a) or (b) only."
            reqs.append({"item_id": r["item_id"], "condition": "t0",
                         "text": text, "image": r.get("image_abs"),
                         "gold": r["answer"].strip("()").upper(),
                         "qtype": "multi-choice", "question": r["question"]})
    elif name == "vstar":
        for r in rows:
            reqs.append({"item_id": r["item_id"], "condition": "t0",
                         "text": r["question"], "image": r.get("image_abs"),
                         "gold": r["answer"].strip().upper(),
                         "qtype": "multi-choice", "question": r["question"]})
    else:
        raise ValueError(f"未知 benchmark: {name}")
    return reqs


def run(benchmarks: list[str], out_dir: Path, limit: int | None,
        workers: int = WORKERS, rpm: int = RPM, model_override: str | None = None) -> None:
    base, key, model = load_env()
    if model_override:
        model = model_override
        out_dir = out_dir  # 调用方负责命名
    global WORKERS_ACTIVE
    WORKERS_ACTIVE = workers
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done: set[tuple[str, str, str]] = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["benchmark"], r["item_id"], r["condition"]))
        logger.info("断点续跑:已有 %d 条", len(done))
    config = {
        "exp": EXP, "model": model, "base_url": base, "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS, "rpm": RPM, "workers": WORKERS, "max_pixels": MAX_PIXELS,
        "image_policy": "native-bytes ≤8MB else downscale",
        "seed": 42, "seed_note": "seed 仅对本地 numpy/torch 生效;API 行不可复现(temperature=0 仍有抖动)",
        "git_commit": git_commit(),
        "runner_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_sha256": __import__("hashlib").sha256(
            Path(__file__).parent.joinpath("scoring.py").read_bytes()).hexdigest(),
        "benchmarks": benchmarks,
        "subsets_sha256": {n: json.loads((BENCH / n / "subset_manifest.json").read_text())["subset_sha256"]
                           for n in benchmarks},
        "started": datetime.now(timezone.utc).isoformat(),
    }
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():  # resume 不覆盖原 config(审查 L1)
        cfg_path = out_dir / f"config_resume_{datetime.now().strftime('%H%M%S')}.json"
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    import subprocess as _sp
    _sp.run([sys.executable, "-m", "pip", "freeze"], stdout=open(out_dir / "requirements.txt", "w"),
            check=False)

    limiter = RateLimiter(rpm)
    lock = threading.Lock()
    fout = results_path.open("a", encoding="utf-8")
    stats: dict[str, list[int]] = {}

    def worker(bench: str, req: dict[str, Any]) -> dict[str, Any]:
        resp = call_flash(base, key, model, req["text"], req["image"], limiter)
        if bench == "mathverse":
            s = score_mathverse(resp["content"], req["question"], req["gold"],
                                req["qtype"], use_judge=False)
            answer, correct, method = s["answer"], s["correct"], s["method"]
            needs_judge = (req["qtype"] == "free-form" and not correct
                           and resp.get("finish_reason") == "stop")
        else:
            letters = valid_letters(req.get("question", ""))
            answer, method = extract_option_letter(resp["content"], letters)
            correct = bool(answer and answer.upper() == req["gold"])
            needs_judge = False
        return {
            "exp": EXP, "model": model, "benchmark": bench,
            "item_id": req["item_id"], "condition": req["condition"],
            "step_t": 0, "t_c": None,
            "answer": answer, "correct": correct,
            "delta_logp": None, "flip": None, "n_tool_calls": None, "coverage": None,
            "seed": 42, "ts": datetime.now(timezone.utc).isoformat(),
            "raw_answer": resp["content"], "extract_method": method,
            "latency_s": resp.get("latency"), "tokens_out": resp.get("tokens_out"),
            "finish_reason": resp.get("finish_reason"),
            "reasoning_tokens": resp.get("reasoning_tokens"),
            "attempts": resp.get("attempts"), "needs_judge": needs_judge,
            "error": resp.get("error"),
        }

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for bench in benchmarks:
            reqs = build_requests(bench)
            if limit:
                reqs = reqs[:limit]
            for req in reqs:
                if (bench, req["item_id"], req["condition"]) in done:
                    continue
                futures.append(ex.submit(worker, bench, req))
            logger.info("%s: %d 请求入队(累计 pending %d)", bench, len(reqs), len(futures))
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                s = stats.setdefault(rec["benchmark"], [0, 0])
                s[0] += int(rec["correct"]); s[1] += 1
            if i % 25 == 0:
                logger.info("进度 %d: %s", i, {k: f"{c}/{n}" for k, (c, n) in stats.items()})
    fout.close()
    logger.info("完成: %s", {k: f"{c}/{n}={c/n:.1%}" if n else "-" for k, (c, n) in stats.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default="mmvp,vstar,mmstar,hr_bench,mathverse")
    ap.add_argument("--limit", type=int, default=None, help="每个 benchmark 取前 N 题(试点)")
    ap.add_argument("--out", default=None, help="run 目录(默认 obs/runs/api_t0_flash_<ts>)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--rpm", type=int, default=RPM)
    ap.add_argument("--model", default=None, help="覆盖 LITELLM_SMOKE_MODEL")
    args = ap.parse_args()
    benchmarks = args.benchmarks.split(",")
    out_dir = Path(args.out) if args.out else REPO / "obs" / "runs" / f"api_t0_flash_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run(benchmarks, out_dir, args.limit, workers=args.workers, rpm=args.rpm, model_override=args.model)
    print(f"run dir: {out_dir}")


if __name__ == "__main__":
    main()
