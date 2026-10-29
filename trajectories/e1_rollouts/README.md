# E1 rollouts（552 题 × 4 次采样，每个条件 2,208 条）

每个条件一个 gzip JSONL，一行一条 rollout，助手各轮原文照录。生成脚本：[`pipeline/eval/baselines/export_e1_rollouts.py`](../../pipeline/eval/baselines/export_e1_rollouts.py)；评分与 [`summarize.py`](../../pipeline/eval/baselines/summarize.py) 相同，`correct` 的均值与 [`summary.json`](../../logs/baselines_v52_20261024/summary.json) 的 `acc` 逐条件一致（导出时断言）。条件说明见 [docs/01 §5.1](../../docs/01_overview.md)。

| 文件 | 条件 | 说明 | rollout | 准确率 | 裁图总数 |
|---|---|---|---:|---:|---:|
| `B0.jsonl.gz` | 无 SFT（无工具） | 原模型 不带工具 | 2,208 | 0.5217 | 0 |
| `B1.jsonl.gz` | 无 SFT（原生 zoom） | 原模型 原生 zoom 工具 | 2,208 | 0.5888 | 3,747 |
| `B2.jsonl.gz` | 无 SFT | 原模型 caption zero-shot | 2,208 | 0.0068 | 197 |
| `B3.jsonl.gz` | SFT | A-SFT | 2,208 | 0.5670 | 2,195 |
| `H1.jsonl.gz` | 答案式 SFT | 答案式 SFT | 2,208 | 0.4515 | 0 |
| `H2.jsonl.gz` | 答案式 SFT + GRPO（旧） | 答案式 SFT + GRPO | 2,208 | 0.4792 | 0 |
| `B6.jsonl.gz` | 无 SFT + GRPO（无工具） | 原模型 + GRPO 无工具（6k，第 47 步） | 2,208 | 0.5960 | 0 |
| `X1.jsonl.gz` | Mini-o3-7B | Mini-o3-7B-v1（官方协议） | 2,208 | 0.6395 | 5,969 |
| `X2.jsonl.gz` | DeepEyes-7B | DeepEyes-7B（官方协议） | 2,208 | 0.5589 | 1,882 |
| `B5.jsonl.gz` | 无 SFT + GRPO | 原模型 + caption GRPO（6k，第 47 步） | 2,208 | 0.0077 | 326 |
| `H2-6k.jsonl.gz` | 答案式 SFT + GRPO | 答案式 SFT + 答案式 GRPO（6k，第 47 步） | 2,208 | 0.5041 | 0 |
| `B4.jsonl.gz` | SFT + GRPO | SFT + GRPO（6k，第 47 步） | 2,208 | 0.6603 | 5,257 |
| `B0-ans.jsonl.gz` | 无 SFT（直接作答） | 原模型 直接作答（无推理指令） | 2,208 | 0.4094 | 0 |
| `B3-cot.jsonl.gz` | SFT（CoT） | A-SFT CoT 提示（无工具） | 2,208 | 0.5380 | 0 |
| `B3-ans.jsonl.gz` | SFT（直接作答） | A-SFT 直接作答（无推理指令） | 2,208 | 0.3863 | 0 |
| `B3-zoom.jsonl.gz` | SFT（原生 zoom） | A-SFT 原生 zoom 工具 | 2,208 | 0.3216 | 12,286 |
| `K-GRPO.jsonl.gz` | SFT + GRPO（4k×8） | SFT + GRPO（4k×8，第 32 步） | 2,208 | 0.6390 | 4,207 |
| `K-DAPO.jsonl.gz` | SFT + DAPO（4k×8） | SFT + DAPO（4k×8，第 32 步） | 2,208 | 0.6635 | 4,620 |
| `K-GSPO.jsonl.gz` | SFT + GSPO（4k×8） | SFT + GSPO（4k×8，第 32 步） | 2,208 | 0.6395 | 4,997 |
| `K-GiGPO.jsonl.gz` | SFT + GiGPO（4k×8） | SFT + GiGPO（4k×8，第 32 步） | 2,208 | 0.6209 | 4,412 |
| `K-base.jsonl.gz` | 无 SFT + GRPO（4k×8） | 原模型 + caption GRPO（4k×8，第 32 步） | 2,208 | 0.0054 | 254 |

## 字段

| 字段 | 说明 |
|---|---|
| `condition` | 条件 ID（上表文件名） |
| `bench`, `item_id` | 题源与我们题池的题号（重新挂题方法见 [trajectories/README](../README.md)） |
| `sample_idx` | 第几次采样（0–3） |
| `gold_letter` | 标准答案选项 |
| `final_answer`, `has_answer_tag` | 评分脚本从最后一轮抽到的选项；是否有 `<answer>X</answer>` 标签 |
| `correct` | 评分脚本判对（截断或出错计 0） |
| `correct_train_rule` | 训练奖励规则（任一轮协议违规或截断即 0、全程恰好一个 `<answer>`）；只有 caption 协议条件有 |
| `steps`, `crops`, `boxes_abs`, `image_wh` | 助手轮数、执行的裁图数、裁图框（原图像素）、原图尺寸 |
| `end`, `truncated`, `error` | 结束方式、是否截断、报错 |
| `prompt_tokens`, `gen_tokens` | 每轮完整输入累计、生成累计 |
| `turns[]` | 助手轮：`step`、`text`、`finish`、`n_tokens`；caption 协议另有 `kind`（answer / grounding / null）与 `check_errors`；原生工具条件有 `tools` / `tool`；部分条件有 `box_abs` |

不含：题目与选项原文、图像、工具回执图像、机器路径。助手原文中复述题干首行（≥ 40 字符）的地方替换为 `[question stem removed]`；DeepEyes-7B（X2）在 `</tool_call>` 之后自己写出的假图片链接、`data:image` 与 base64 串替换为 `[removed:<类型>]`。每个条件的替换次数见 [`STATS.json`](STATS.json) 的 `redacted_in_assistant_text`。
