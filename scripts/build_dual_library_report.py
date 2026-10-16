"""Render docs/22 (dual-library report) to the Pages HTML, so the two never drift.

Usage: python scripts/build_dual_library_report.py [--check]
"""
import argparse
import html
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "22_dual_library_eval_20261016.md"
CSS = ROOT / "scripts" / "report_template.css"
OUT = ROOT / "web" / "reports" / "dual_library_eval_20261016.html"
REPO = "https://github.com/CrepuscularIRIS/vlm-caption-trajectories"


LIST_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s")


def normalize_lists(md_text: str) -> str:
    """Adapt GitHub-style lists to Python-Markdown: blank line before a list, 4-space nesting."""
    out: list[str] = []
    prev = ""
    fenced = False
    for line in md_text.splitlines():
        quote = "> " if line.startswith("> ") else ""
        text = line[len(quote):]
        if text.startswith("```"):
            fenced = not fenced
        item = None if fenced else LIST_ITEM.match(text)
        if item:
            indent = len(item.group(1))
            text = " " * (4 * ((indent + 1) // 3)) + text.lstrip()
            prev_text = prev[len(quote):] if prev.startswith(quote) else prev
            if prev_text.strip() and not LIST_ITEM.match(prev_text) and not prev_text.startswith(" "):
                out.append(quote.rstrip())
        out.append(quote + text)
        prev = line
    return "\n".join(out)


def render(md_text: str) -> str:
    lines = normalize_lists(md_text).splitlines()
    title = lines[0].lstrip("# ").strip()
    body_md = "\n".join(lines[1:])
    conv = markdown.Markdown(extensions=["tables", "fenced_code", "toc"], extension_configs={"toc": {"toc_depth": "2"}})
    body = conv.convert(body_md)
    body = re.sub(r"<table>", '<div class="tbl"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)
    body = re.sub(r'<h2 id="([^"]+)">(\d+) ', r'<h2 id="\1"><span class="n">\2</span>', body)
    short = re.sub(r"（.*?）", "", title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>双库轨迹评估汇报</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
{CSS.read_text()}</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow">导师汇报 · 2026-10-16 · Qwen3-VL-4B · SFT → GRPO</div>
  <h1>{html.escape(short)}</h1>
</header>
<nav class="toc">{conv.toc}</nav>
{body}
<footer>由 <code>docs/22_dual_library_eval_20261016.md</code> 生成（<code>scripts/build_dual_library_report.py</code>）。
<a href="{REPO}/blob/main/docs/22_dual_library_eval_20261016.md">Markdown 原文</a> · <a href="{REPO}">仓库</a></footer>
</div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the HTML is stale")
    out = render(SRC.read_text())
    if parser.parse_args().check:
        if OUT.read_text() != out:
            sys.exit(f"{OUT.relative_to(ROOT)} is stale; rerun without --check")
        return
    OUT.write_text(out)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(out):,} bytes)")


if __name__ == "__main__":
    main()
