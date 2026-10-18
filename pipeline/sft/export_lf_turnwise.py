"""Turn-wise export of the v4 SFT files for LLaMA-Factory (per-turn loss mask without trainer changes).

为什么需要:v4 的 `assistant_supervision` 标出哪些 assistant 轮计 loss(被强制失误的轮为 False)。
LLaMA-Factory 的 sharegpt 没有逐轮掩码,但有 `mask_history: true`(只对最后一个 assistant 轮计 loss)。
本脚本把每条轨迹按"每个计 loss 的轮一条样本"展开:样本 = 该轮之前的完整对话(原样,严格
human/gpt 交替,失误轮仍是 gpt 轮) + 该轮本身。配合 `mask_history: true`,得到的监督 token 与
逐轮掩码完全相同,不合并、不折叠任何消息。

用法(仓库根目录):
    python3 scripts/export_lf_turnwise.py            # 生成并校验两个 *_turnwise.json
    python3 scripts/export_lf_turnwise.py --check    # 只校验已生成的文件

训练配置必须包含:
    mask_history: true
    preserve_thinking: true        # 旧版 LLaMA-Factory 无此项时改用 template: qwen3_vl_nothink
    cutoff_len: 8192               # 最长样本 7,563 token(Qwen3-VL-8B 处理器);截断会丢历史视图
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SOURCES = ['caption_sft_v4_interp_train', 'caption_sft_v4_interp_train_all_tasks', 'caption_sft_v4c_interp_train']
EXPECTED = DATA / 'interp_v4' / 'LF_TURNWISE.json'


def expand(row: dict) -> list[dict]:
    """One sample per supervised assistant turn; history is copied verbatim."""
    conv, mask = row['conversations'], row['assistant_supervision']
    head = conv[:1] if conv[0]['from'] == 'system' else []
    body = conv[len(head):]
    if len(body) % 2 or len(mask) != len(body) // 2:
        raise ValueError(f"{row['id']}: turns and supervision mask disagree")
    out = []
    for turn, supervised in enumerate(mask):
        user, asst = body[2 * turn], body[2 * turn + 1]
        if user['from'] != 'human' or asst['from'] != 'gpt':
            raise ValueError(f"{row['id']}: not strictly alternating at turn {turn + 1}")
        if not supervised:
            continue
        prefix = body[:2 * turn + 2]
        n_images = sum(m['value'].count('<image>') for m in prefix)
        out.append({'id': f"{row['id']}#t{turn + 1}", 'task_id': row['id'], 'target_turn': turn + 1,
                    'conversations': head + prefix, 'images': row['images'][:n_images]})
    return out


def summarize(source: list[dict], rows: list[dict]) -> dict:
    """Counts that must hold for the turn-wise file to be equivalent to the per-turn mask."""
    targets = {(r['task_id'], r['target_turn']): r['conversations'][-1]['value'] for r in rows}
    want = {(s['id'], t + 1): s['conversations'][(1 if s['conversations'][0]['from'] == 'system' else 0) + 2 * t + 1]['value']
            for s in source for t, keep in enumerate(s['assistant_supervision']) if keep}
    if targets != want:
        raise ValueError('turn-wise targets differ from the supervised turns of the source file')
    for r in rows:
        body = r['conversations'][1:] if r['conversations'][0]['from'] == 'system' else r['conversations']
        if len(body) % 2 or any(m['from'] != ('human', 'gpt')[i % 2] for i, m in enumerate(body)):
            raise ValueError(f"{r['id']}: not strictly alternating")
        if sum(m['value'].count('<image>') for m in body) != len(r['images']):
            raise ValueError(f"{r['id']}: image placeholders and image list disagree")
    return {'source_rows': len(source), 'source_assistant_turns': sum(len(s['assistant_supervision']) for s in source),
            'rows': len(rows), 'history_turns_in_context_not_supervised': sum(r['target_turn'] - 1 for r in rows),
            'max_images_per_row': max(len(r['images']) for r in rows)}


def main() -> int:
    check_only = '--check' in sys.argv
    expected = json.loads(EXPECTED.read_text()) if EXPECTED.exists() else {}
    report = {}
    for name in SOURCES:
        source = json.loads((DATA / f'{name}.json').read_text())
        path = DATA / f'{name}_turnwise.json'
        if check_only:
            rows = json.loads(path.read_text())
        else:
            rows = [r for s in source for r in expand(s)]
            path.write_text(json.dumps(rows, ensure_ascii=False, separators=(',', ':')))
        report[path.name] = {**summarize(source, rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print(path.name, json.dumps(report[path.name]))
    if '--write-expected' in sys.argv:
        EXPECTED.write_text(json.dumps(report, indent=1) + '\n')
        return 0
    if expected and report != expected:
        print('MISMATCH against', EXPECTED.relative_to(ROOT))
        return 1
    print('turn-wise export OK' + ('' if expected else ' (no expected record to compare)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
