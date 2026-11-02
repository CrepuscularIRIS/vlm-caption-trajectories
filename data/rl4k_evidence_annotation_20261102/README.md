# rl4k 题池证据标注（L0–L2 + 线索层）

2026-11-02 交付。对象：caption v5.1 协议 RL 训练题池 `grpo_caption_v51_4k` 的 4,096 道多选题。每题标注"答对本题所需/可得的视觉证据"，分三层；答案无法从图核验的题另标线索层。**不含题面与图像**，记录按 uid 索引，坐标为相对原图的 [0,1] 归一化。规范见 [vlm-rl-signal-design/design/ANNOTATION_SPEC.md](https://github.com/CrepuscularIRIS/vlm-rl-signal-design/blob/main/design/ANNOTATION_SPEC.md)（v2）与同库 CLUE_ANNOTATION_AND_CONTROLLED_RUNS.md（线索层）。

## 流程（全自动管线，可复算）

1. gpt-6-astra 经代理按规范两轮标注（草稿 → 叠框图+原分辨率裁图核对，校验错误回灌；`scripts` 见 vlm-rl-signal-design）；
2. Claude 子代理按审查图逐题审查并改正（审查记录在 `reviews/`）；
3. `needs_external_knowledge` 215 题按研究者决定重标为可回答（图是必要证据）；
4. 改正按 rv→smp→reanno 顺序覆盖合并为终稿，全量校验 0 error。

## 文件与计数

| 文件 | 行数 | 内容 |
|---|---:|---|
| `L0/batch.jsonl` | 4,096 | unit（9 类）、answerability（4 类）、gt_check、required_tier、task_tags |
| `L1/batch.jsonl` | 3,559 | 带 id/role 的证据框、负候选、搜索范围、主图区、归属上下文 |
| `L2/batch.jsonl` | 3,255 | 转录值（读不清记 `?`）、绑定元组（table/chart/entity/relation/count/compare/chain）、充分证据组合、区分条件 |
| `lc/batch.jsonl` | 689 | 线索层：537 题止于 L0 的（可见线索框→可支持推测→未解决；歧义题多解；evidence_not_found 记搜索范围）+ 152 题 gt 存疑的读数结论（`reading_answer`，只记录不改 gt） |
| `reviews/v2_review.jsonl` | 1,933 | 逐题审查 verdict 与改动记录（pass 1,499 / 小改 90 / 大改 268 / 重标 215 内含） |
| `reviews/lc_review.jsonl` | 25 | 线索层抽查 |
| `gt_disagree.tsv` | 152 | 看图与 gt 不符清单（uid/来源/unit/gt字母/依据） |

终稿分布：unit single_box 1,612 / keyed_set 1,235 / unanswerable 537 / region_global 254 / multi_set 185 / multi_compare 139 / pair_relation 74 / absent 50 / ordered_chain 10。answerability：可回答 3,509 / 题目不可答 508 / 目标不存在 50 / 分辨率不足 29。gt_check：agree 3,369 / cannot_tell 575 / disagree 152。

## 已知边界

- 152 题 gt 存疑只记录未修改；其中 6 条读数结论与自身说明矛盾，已记 `none`。
- gt 存疑题与 E1 评测池零重叠；对 k8 训练奖励为共模噪声（±1–3 点内，不改算法排序）。
- 线索框只作审计参照，不指定模型动作，未进入任何 reward。

## 哈希
bf5252daa81712a6410a20cd721f2fd55be74a943e326f67f127501ad92258bb  data/rl4k_evidence_annotation_20261102/L0/batch.jsonl
f8eb319c53d167b35328229bd80e1d1b8efc40ac8dc7a8dc7adc56c7169ea87e  data/rl4k_evidence_annotation_20261102/L1/batch.jsonl
3c940c635a578d2aa9658645f838636af48468318e3c92ccdcf8e82f2f218b00  data/rl4k_evidence_annotation_20261102/L2/batch.jsonl
425abc35843a4bf0aae20bc6a955a31d027d296c77be316e9254fb3acfa7587e  data/rl4k_evidence_annotation_20261102/lc/batch.jsonl
