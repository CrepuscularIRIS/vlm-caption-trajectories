# 第二次分段价值诊断（修测量后）：逐次结果记录

对应 docs/36 第7节、Open Scientist 第019–021步。13例，P0/P1t/P1m/P2t 各16次（u33无P1m），共816行，0次参数更新。

- `runs_sft.jsonl`：7例，448行；`runs_r1.jsonl`：6例，368行。
- `summary.json`：每例每前缀的 V、R、平均裁图与生成 token，以及 core/mixed 的 S1、S2(m)、S3 bootstrap 区间。

每行字段：`id/group/policy/prefix/j` 标识一次测量；`y` 终答是否正确；`crops` 含前缀已有裁图；`r=y*(1-.01*crops)`；`gen_tokens` 新增生成 token 数；`partial_tokens` 注入的助手前缀 token 数。不含题面、图像或生成文本。

| 文件 | SHA-256 |
|---|---|
| runs_sft.jsonl | `8d2c11dcbf506bff1f042fc8ae4696272e76d9e76efd54c3013019e61ed2f3b6` |
| runs_r1.jsonl | `f6be2defca3e8dc6a272033d1f0a51e52ec962ba339fb1154bbcb581fef6d860` |

复算：`work=$(mktemp -d); cp logs/mve/segval_v2/runs_*.jsonl "$work/"; python3 pipeline/eval/mve/summarize_segment_v2.py "$work"`。
