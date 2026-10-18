"""Build a visual review queue; export only hash-bound, explicitly human KEEP entries."""
import argparse
import hashlib
import html
import json
from pathlib import Path

from tiered_transport import write_json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(out):
    review = out / 'selection'; review.mkdir(parents=True, exist_ok=True)
    candidates = []
    sections = []
    for path in sorted((out / 'audit').glob('*.json')):
        a = json.loads(path.read_text())
        if a['decision'] != 'AUTO_PASS_PENDING_MANUAL': continue
        rp = Path(a['result_path']); r = json.loads(rp.read_text()); ro = r['rollout']
        entry = dict(item_id=a['item_id'], group_id=a['group_id'], role=a['selected_role'],
                     result_path=str(rp), result_sha256=sha(rp), audit_path=str(path), audit_sha256=sha(path),
                     verdict='PENDING', reviewer_type=None, reviewer=None, note='')
        candidates.append(entry)
        by_view = {v['view_id']: v for v in ro['views']}
        steps = []
        for s in ro['steps']:
            images = ''.join('<figure><figcaption>' + html.escape(v) + '</figcaption><img loading="lazy" src="' +
                             html.escape(Path(by_view[v]['path']).as_uri(), quote=True) + '"></figure>' for v in s['available_before'])
            steps.append(f'<details><summary>Step {s["step"]} · 已收到的视图</summary>{images}</details><pre>{html.escape(s["raw_text"])}</pre>')
        sections.append('<section><h2>' + html.escape(a['item_id'] + ' / ' + a['selected_role']) + '</h2><pre>' +
                        html.escape(r['item']['question']) + '</pre><p>自动审核通过，人工终裁待完成。</p>' + ''.join(steps) + '</section>')
    write_json(review / 'candidates.json', candidates)
    # Do not overwrite a user's actual verdicts when refreshing the queue.
    template = review / 'manual_template.json'
    if not template.exists(): write_json(template, candidates)
    page = '<!doctype html><meta charset="utf-8"><title>Binding unit visual review</title><style>body{max-width:1100px;margin:32px auto;font:16px sans-serif}pre{white-space:pre-wrap}img{max-width:100%}section{border-top:2px solid #888;padding-top:12px;margin-top:32px}summary{cursor:pointer}</style><h1>实例绑定轨迹 · 看图终裁队列</h1><p>检查每一步当时的证据、对象属性绑定、无据断言、HOLD/REVISE 与动作必要性。PENDING 不导出。</p>' + ''.join(sections)
    (review / 'CASEBOOK.html').write_text(page)
    return candidates


def export(out, manual):
    approved = json.loads(Path(manual).read_text())
    current = {x['item_id']: x for x in build(out)}
    rows, groups, accepted = [], set(), []
    for m in approved:
        if m.get('verdict') != 'KEEP': continue
        if m.get('reviewer_type') != 'human' or not m.get('reviewer') or not m.get('note'):
            raise ValueError('Human KEEP requires reviewer identity and visual evidence note')
        c = current.get(m['item_id'])
        if not c or any(m.get(k) != c[k] for k in ('result_sha256', 'audit_sha256', 'role', 'group_id')):
            raise ValueError('Manual verdict refers to stale or unapproved candidate')
        if c['group_id'] in groups: raise ValueError('Duplicate image group in export')
        r = json.loads(Path(c['result_path']).read_text())
        sample = r['rollout']['export']
        if not sample: raise ValueError('Missing student transcript')
        if 'demo_original_image' in json.dumps(sample): raise ValueError('Demonstration leaked into export')
        rows.append(sample); groups.add(c['group_id']); accepted.append(m)
    write_json(out / 'selection' / 'clean_sft.json', rows)
    write_json(out / 'selection' / 'EXPORT_MANIFEST.json', dict(count=len(rows), manual_sha256=sha(manual),
               selection='single_teacher_independent_review_plus_human_keep', accepted=accepted,
               limitation='No training or LLaMA-Factory preprocessing of this new unit has been run.'))
    return len(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True, type=Path); p.add_argument('--manual', type=Path)
    a = p.parse_args(); a.out = a.out.resolve()
    print(json.dumps(dict(exported=export(a.out, a.manual)) if a.manual else dict(pending_visual_review=len(build(a.out)))))
