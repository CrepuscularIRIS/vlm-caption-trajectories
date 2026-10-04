#!/usr/bin/env python3
"""Render curated public cases or a local visual book; never call a model/API.

Local-only MME views are resolved under --workspace-root and hash-checked before
copying to the selected output directory. That output is not for redistribution.
"""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cases():
    return [json.loads(p.read_text()) for p in sorted((ROOT / 'cases/20261004').glob('*.json'))]


def markdown(data):
    out = ['# 9 个真实案例：好的示范、过程缺陷和共同分歧', '',
        '**9 道题、18 条完整可见轨迹；定向选择，不是随机抽样，也不是质量通过率估计。** '
        '状态取原始审核记录，Codex 的看图判断单列，未填写人工终裁。', '',
        '建议先读 C01 → C02 → C04 → C05 → C07 → C09。'
        '对照 [当前统计](09_production_review_20261004.md) 和 [给导师的问题](10_mentor_questions_20261004.md)。', '',
        'WorldBench 案例附 8 个去重后的真实输入视图；MME 案例公开文字与动作，图片仅在本地查看，'
        '见 [来源、许可与本地案例册](11_publication_and_reproduction_20261004.md)。'
        '所有 caption 均来自保存的可见输出，不包含 API 私有推理；旧协议的 `<think>` 标签也只用于这段显式证据 caption。', '',
        '|案例|要讨论的特点|生成轨迹 / 家族|', '|---|---|---|']
    for c in data:
        out.append(f"|[{c['case_id']} · {c['item_id']}](#{c['case_id'].lower()})|{c['title']}|{len(c['trajectories'])} / {c['independent_generator_families']}|")
    for c in data:
        out += ['', f'<a id="{c["case_id"].lower()}"></a>', '', f"## {c['case_id']} · {c['title']}", '',
            f"**题目 ID：** `{c['item_id']}` · **编辑归类：** `{c['category']}`", '',
            '**任务（中文转述）：** ' + c['task_paraphrase_zh'], '',
            '**观察与边界：** ' + c['editorial_observation_zh'], '',
            '**请导师判断：** ' + c['mentor_question_zh'], '',
            '|生成器|档位|终答 / 原金标|原评分|步数 / crop|原自动审核|', '|---|---|---|---|---|---|']
        for t in c['trajectories']:
            status = t['audit']['decision'] if t['audit'] else '未列入本次过程审核展示'
            out.append(f"|{t['model']}|{t['effort']}|{t['answer']} / {t['gold']}|{t['score']['score']}|{len(t['steps'])} / {t['crops']}|{status}|")
        out += ['', '**人工终裁：未完成。** 本页意见不是人工 KEEP；原金标和原评分没有修改。']
        if c['source_question']:
            out += ['', '<details><summary>WorldBench 原题（原样保留）</summary>', '', '```text', c['source_question'], '```', '', '</details>']
        if not c['images_public']:
            out += ['', '> MME-RealWorld 原图/crop 不随公开仓库分发。本地案例册按相同步骤展示图片，并核对输入视图哈希。']
        shown = set()
        for t in c['trajectories']:
            out += ['', f"### {t['model']} · {len(t['steps'])} 步", '']
            views = {v['view_id']: v for v in t['views']}

            def show(vid):
                v = views[vid]
                asset = v['public_asset']
                if asset and asset not in shown:
                    shown.add(asset)
                    return ['', f"![{c['case_id']} {t['model']} {vid}](../{asset})", '',
                            f"*真实收到的 `{vid}`；完整哈希与来源见 JSON。*", '']
                if asset:
                    return ['', f"视图 `{vid}` 与本案例上方已显示图像相同：[查看](../{asset})。", '']
                return ['', f"视图 `{vid}`：本地可查，SHA-256 `{v['sha256']}`。", '']

            out += show('original_image')
            for step in t['steps']:
                out += [f"**Step {step['step']} · {step['behavior']}**。此前已收到：`{'`, `'.join(step['available_before'])}`。", '',
                        '```text', step['visible_output'], '```']
                if step['observation']:
                    out += show(step['observation'])
            if t['audit']:
                a = t['audit']
                out += ['', f"审核原因：`{json.dumps(a.get('reasons', []), ensure_ascii=False)}`；"
                    f"fallback_unreviewed：`{a.get('fallback_unreviewed', False)}`；"
                    f"终答盲读视图：`{json.dumps(a.get('final_blind', {}).get('views', []))}`。"]
            out += ['', '<details><summary>可追溯信息</summary>', '',
                f"- 原结果相对路径：`{t['source_path']}`", f"- 原结果 SHA-256：`{t['source_sha256']}`",
                '- 这里展示的是保存的可见 turn；生成器可能做过标签归一，记录保留在案例 JSON。', '', '</details>']
        out += ['', f"[完整案例 JSON](../cases/20261004/{c['case_id']}_{c['item_id']}.json)"]
    out += ['', '## 如何使用这些特例', '',
        'C01–C03 用来讨论目标行为；C04–C06 检查行为、前缀和审核门的边界；C07–C09 区分共同盲点、范围歧义和金标待核。'
        '**不要把全部案例直接作为 SFT 正例，不要用本页推断全池失败率。**', '',
        '图片归属与修改说明见 [ATTRIBUTION](../media/20261004/ATTRIBUTION.json)。']
    return '\n'.join(out) + '\n'


def render_local(data, workspace, output):
    workspace, output = workspace.resolve(), output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Local visual book must be outside the public repository')
    output.parent.mkdir(parents=True, exist_ok=True)
    media = output.parent / 'views'
    media.mkdir(exist_ok=True)
    assets = {}
    for c in data:
        for t in c['trajectories']:
            for v in t['views']:
                src = ((ROOT / v['public_asset']) if v['public_asset'] else (workspace / v['source_path'])).resolve()
                allowed = ROOT if v['public_asset'] else workspace
                if not src.is_relative_to(allowed):
                    raise ValueError('View path outside allowed root')
                blob = src.read_bytes()
                if hashlib.sha256(blob).hexdigest() != v['sha256']:
                    raise ValueError('View hash mismatch: ' + str(src))
                ext = '.jpg' if blob.startswith(b'\xff\xd8') else '.png'
                dest = media / (v['sha256'] + ext)
                shutil.copyfile(src, dest)
                assets[v['sha256']] = 'views/' + dest.name
    esc = html.escape
    h = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>轨迹案例册 · 2026-10-04</title>',
         '<style>body{font:17px/1.65 system-ui,sans-serif;max-width:1120px;margin:auto;padding:30px;color:#202d3a;background:#f5f7fa}'
         'article{background:white;border:1px solid #d8e1ea;border-radius:12px;padding:26px;margin:30px 0}'
         'img{display:block;max-width:100%;max-height:720px;object-fit:contain;margin:15px auto}'
         'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f6;padding:16px;font-size:14px}'
         'a{color:#155fa0}h2{border-bottom:2px solid #3980a5;padding-bottom:10px}.note{background:#fff1cd;padding:16px}'
         'small{overflow-wrap:anywhere}@media print{body{background:white}article{break-before:page}details{display:block}}</style>',
         '<h1>9 个真实轨迹案例 · 本地完整图版</h1><p class="note">仅供本地研究审阅，含未获再分发许可的 MME 原图与 crop。'
         '不要把本目录整体上传公开仓库。全部人工终裁仍待填写；Codex 意见不等于人工 KEEP。</p>',
         '<p>定向选择 9 题、18 条轨迹。步骤按实际时序展示，不能用后来的图像替早期错误补证。</p>', '<nav>']
    h += [f'<a href="#{c["case_id"]}">{c["case_id"]} {esc(c["title"])}</a><br>' for c in data]
    h += ['</nav>']
    for c in data:
        h += [f'<article id="{c["case_id"]}"><h2>{c["case_id"]} · {esc(c["title"])}</h2>',
              f'<p><b>{esc(c["item_id"])}</b> · {esc(c["task_paraphrase_zh"])}</p>',
              f'<p>{esc(c["editorial_observation_zh"])}</p>', f'<p class="note">请导师判断：{esc(c["mentor_question_zh"])}</p>']
        for t in c['trajectories']:
            a = t['audit'] or {}
            h += [f'<h3>{esc(t["model"])} · {esc(str(t["effort"]))}</h3>',
                  f'<p>终答 / 原金标：{esc(str(t["answer"]))} / {esc(str(t["gold"]))}；'
                  f'原评分：{t["score"]["score"]}；原审核：{esc(a.get("decision", "未列入本次过程审核展示"))}</p>']
            views = {v['view_id']:v for v in t['views']}

            def view_html(vid):
                v = views[vid]
                return f'<figure><img loading="lazy" src="{assets[v["sha256"]]}" alt="{esc(vid)}"><figcaption>收到 {esc(vid)} · <small>{v["sha256"]}</small></figcaption></figure>'

            h += [view_html('original_image')]
            for s in t['steps']:
                h += [f'<p><b>Step {s["step"]}</b> · 此前可用：{esc(", ".join(s["available_before"]))}</p>',
                      '<pre>' + esc(s['visible_output']) + '</pre>']
                if s['observation']:
                    h += [view_html(s['observation'])]
            h += ['<details><summary>原审核状态与来源</summary><pre>' + esc(json.dumps(a,ensure_ascii=False,indent=2)) + '</pre>',
                  '<small>' + esc(t['source_path'] + '\nSHA-256: ' + t['source_sha256']) + '</small></details>']
        h += ['</article>']
    h += ['<p>WorldBench 图片按其数据卡 CC-BY-4.0 归属；MME 图片仅本地查看。原图/金标未修改。</p></html>']
    output.write_text('\n'.join(h))
    print(json.dumps({'local_casebook':str(output),'cases':len(data),'unique_views':len(assets),'view_hashes':'PASS'}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-markdown', action='store_true')
    ap.add_argument('--write-markdown', action='store_true')
    ap.add_argument('--workspace-root', type=Path)
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    data = cases()
    target = ROOT / 'docs/CASEBOOK_20261004.md'
    if a.write_markdown:
        target.write_text(markdown(data))
    if a.check_markdown:
        if target.read_text() != markdown(data):
            raise ValueError('Casebook markdown differs from published cases')
        print('Casebook text consistency: PASS (not visual acceptance)')
    if a.workspace_root and a.output:
        render_local(data, a.workspace_root, a.output)
    elif a.workspace_root or a.output:
        ap.error('--workspace-root and --output must be used together')
    elif not (a.write_markdown or a.check_markdown):
        ap.error('Choose a markdown action or local output')


if __name__ == '__main__':
    main()
