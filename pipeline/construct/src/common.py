#!/usr/bin/env python3
"""Shared experiment helpers (Handoff_observe.md §1).

用法:
    from common import set_seed, log_environment, load_subset, git_commit
    set_seed(42); rows = load_subset("mathverse")   # rows[i]["image_abs"] 为绝对路径
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]  # ${ICML_DIR}
BENCH = REPO / "obs" / "benchmarks"
REQUIRED_FIELDS = ("item_id", "image_path", "question", "answer", "label", "exps")

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility (experiment-reproducibility rule)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_environment() -> dict[str, Any]:
    """Record environment information for reproducibility."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
    }


def git_commit() -> str | None:
    """HEAD commit of this repo, or None when not a git repo."""
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def norm_value(s: str) -> str:
    """归一化数值/单位文本:'$140^\\circ$'→'140°','80.0'→'80'。"""
    s = s.replace("\\circ", "°").replace("$", "").replace(" ", "").replace(",", "")
    s = re.sub(r"\.0+(?=[^\d]|$)", "", s)
    return s.lower().rstrip(".")


def to_float(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def final_value(content_tail: str) -> str | None:
    """从回答尾部抽最终值:'…is $140^\\circ$'→'140°'。"""
    lines = [l for l in (content_tail or "").strip().splitlines() if l.strip()]
    tail = " ".join(lines[-3:])[-160:] if lines else ""
    if not tail:
        return None
    m = list(re.finditer(r"(?:is|:|=)\s*\*{0,2}\$?([^$*\n]{1,40}?)\$?\*{0,2}(?:[.,;]?\s*$|\n)", tail))
    if m:
        v = m[-1].group(1).strip()
        if v:
            return norm_value(v)
    v = tail.strip().split()[-1] if tail.strip() else ""
    return norm_value(v) if v else None


def value_eq(ans: str | None, gold: str) -> bool:
    """答案值与 gold 等价判定(数值严格相等,或 len≥2 的包含)。"""
    if not ans:
        return False
    a, g = norm_value(ans), norm_value(gold)
    fa, fg = to_float(a), to_float(g)
    if fa is not None and fg is not None:
        return abs(fa - fg) < 1e-6
    return len(a) >= 2 and (a in g or g in a)


def load_subset(name: str, check_images: bool = True) -> list[dict[str, Any]]:
    """Load obs/benchmarks/<name>/subset.jsonl with schema and path validation.

    - asserts Handoff §1 fixed fields exist on every row
    - asserts item_id uniqueness
    - resolves image_path (relative to obs/benchmarks/<name>/) into image_abs
    """
    path = BENCH / name / "subset.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在 — 先跑 obs/src/make_subsets.py")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"{name}: item_id 不唯一: {dup[:5]}")
    for r in rows:
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            raise ValueError(f"{name}/{r.get('item_id')}: 缺字段 {missing}")
        if check_images and r["image_path"]:
            abs_p = (BENCH / name / r["image_path"]).resolve()
            if not abs_p.exists():
                raise FileNotFoundError(f"{name}/{r['item_id']}: {r['image_path']}")
            r["image_abs"] = str(abs_p)
    logger.info("load_subset(%s): %d rows", name, len(rows))
    return rows
