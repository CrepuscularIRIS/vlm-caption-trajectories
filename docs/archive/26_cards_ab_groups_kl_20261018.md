# 卡 A / 卡 B 中期、rollout 分组统计、KL 与新增基线（2026-10-18 晚）

> 10-19 更新：所有组已跑完，最终表格见 [docs/27](27_results_20261019.md)。本页第 6 节的“待测”行以 docs/27 为准；答案式 SFT 的 .424 作废（导出的 config 缺 `rope_scaling`），修正后为 .458。

状态：卡 A、卡 B 训练中；KL 与对照重跑已排队（服务器自动接续）；两个新基线已立项。本页只记录已有数据与计划，结论待训练完成后补全。

## 1. 卡 A / 卡 B 第 40 步

两组都从库 A 的 SFT（`sft_v22_turnwise`）起跑，165 步，n=16，lr 1e-6，无 KL（与原版 GRPO-A 相同）。验证集：训练内 480 题中的 192 题批，T=0.7 单次采样（与 E1 口径不同，E1 原版 GRPO-A 为 .664）。

| run | 验证准确率 step 0 | step 40 |
|---|---|---|
| 原版 GRPO-A `grpo_v3_A` | .502 | .566 |
| crop-floor λ=0.025 | .502 | .564 |
| crop-floor λ=0.05 | .502 | .591 |
| crop-cost λ=0.01（塌缩，第 56 步停） | .502 | .484 |
| **卡 A** `grpo_v3_A_cc` | .502 | **.589** |
| 卡 B `grpo_v3_A_cc_dsasym` | .502 | 待出 |

训练 rollout 第 34–40 步：卡 A 每题裁图 1.74、直答 7%、通过验证的裁图 19%；crop-floor λ=0.025 为 1.64 / 7%。裁图数在所有 run 中都随训练上升，卡 A 尚未压低裁图。逐步指标：[`logs/grpo/grpo_v3_A_cc_metrics.csv`](../../logs/grpo/grpo_v3_A_cc_metrics.csv)、[`logs/grpo/grpo_v3_A_cc_dsasym_metrics.csv`](../../logs/grpo/grpo_v3_A_cc_dsasym_metrics.csv)（训练中导出，非最终）。

## 2. rollout 按组划分：全对 / 全错 / 有对有错

导师要求：把 rollout 按组分成全对、全错、有对有错三类，分别统计直答与裁图的正确率、advantage、监督 token 数。

- 脚本：[`pipeline/grpo/v51_cc/group_stats.py`](../../pipeline/grpo/v51_cc/group_stats.py)，在服务器上对训练 rollout 日志计算，advantage 按各 run 实际公式由记录的分数重算。
- 全部五个 run 的完整表：[`logs/grpo/group_stats_20261018.md`](../../logs/grpo/group_stats_20261018.md)。
- 原版 GRPO-A 当时没有记录 rollout，不在表内；对照组统计由下面第 3 节的对照重跑补上。

卡 A（第 1–50 步，600 组）：

| 组 | 类型 | rollout 数 | 正确率 | 平均 advantage | 平均 \|advantage\| | 每条监督 token | Σ\|adv\|·token 占比 |
|---|---|---|---|---|---|---|---|
| 全对（27%） | 直答 | 484 | 1.00 | −0.009 | 0.025 | 51 | 0.0% |
| | 裁图 | 2,076 | 1.00 | +0.002 | 0.062 | 163 | 1.9% |
| 有对有错（69%） | 直答 | 1,196 | 0.44 | −0.094 | 0.814 | 92 | 6.7% |
| | 裁图 | 5,396 | 0.55 | +0.021 | 0.779 | 259 | 91.4% |
| 全错（5%） | 直答 / 裁图 | 448 | 0 | 0 | 0 | — | 0% |

各 run 有对有错组的对比：

| run | 直答 / 裁图 正确率 | 有对有错组裁图轨迹的 Σ\|adv\|·token 占比 | 全对组的 Σ\|adv\|·token 占比 |
|---|---|---|---|
| 卡 A | .44 / .55 | 91.4% | 1.9% |
| 卡 B | .44 / .57 | 90.4% | 1.5% |
| crop-floor λ=0.025 | .40 / .56 | 93.2% | 1.1% |
| crop-floor λ=0.05 | .42 / .57 | 88.9% | 1.8% |
| crop-cost λ=0.01 | .46 / .63 | 47.1% | 12.0% |

读法：
1. 实际策略更新几乎全部来自有对有错组。其中直答的正确率比裁图低 11–16 个点，平均 advantage 为负，所以结果奖励本身就在把直答往下压，这解释了各 run 的裁图上升。
2. 全错组 advantage 恒为 0，不产生梯度；DS 丢掉它只省计算，不改变更新方向。
3. 全对组的权重只有 1–2%，卡 A 的价格通道也主要落在这里，力度不足以抵消第 1 条。
4. 裁图轨迹的监督 token 是直答的 2.5–3 倍；按 token 平均的损失让长轨迹占更大权重。
5. 更正：crop-floor 两组的 λ 是 0.025 / 0.05（与第 24 号报告一致）。

## 3. KL

- 现状：所有 GRPO run（含原版 GRPO-A）`kl_coef=0.0 disable_kl=true`。10-14 起关闭，原因是系数已为 0 时参考模型仍被加载、每步多算一遍 log-prob；从未做过 KL 对照。
- 导师意见：KL 用于避免策略更新离 SFT 参考太远，一般要加。
- 计划（已排队，卡 A 跑满 165 步后在同两张卡上自动启动）：[`grpo_kl.sh`](../../pipeline/grpo/scripts/grpo_kl.sh) = 原版 GRPO-A 奖励（只看对错）+ `low_var_kl`，`kl_coef=1e-2`（EasyR1 示例默认），记录 rollout。
- 对照重跑（卡 B 跑满后自动启动）：原版 GRPO-A 配置、同种子、记录 rollout，用于补齐对照组的分组统计，并与 KL 组构成同代码的有/无 KL 对比。接续脚本：[`wait_then_kl.sh`](../../pipeline/grpo/scripts/wait_then_kl.sh)、[`wait_then_ctrl.sh`](../../pipeline/grpo/scripts/wait_then_ctrl.sh)。
- 之后：按卡 A / 卡 B 的结果，把较好的一个与 KL 版 GRPO 组合。

## 4. 新增两个基线（导师要求）

| 基线 | 内容 | 状态 |
|---|---|---|
| SFT cold-start w/o RL | 库 A SFT 起点，不做 RL | 已完成：由现有 E1 逐样本输出算出，见第 6 节表 1 |
| 答案式 SFT cold-start w/ RL | 同题同图，SFT 目标只有最终答案（无 caption、无裁图），再做与原版 GRPO-A 相同的 GRPO | 代码与数据已就绪，排队中 |

答案式基线的实现要点：
- 原 GRPO 环境的解析器要求每轮有 `<think>`、`Step N | WORD` 头和 caption；只输出 `<answer>` 会被判无效、奖励为 0。所以新增协议 [`caption_ao.py`](../../pipeline/grpo/v51_ansonly/caption_ao.py)：工具、动作格式、裁图规则、回执与预算 8 都与 v5.1 相同（回执函数直接复用），只去掉 caption 合同；每轮恰好一个动作，可选一个不检查内容的 `<think>`。插件目录其余文件与 `v51_cropfloor` 相同，只把协议模块换成 `caption_ao`。
- 数据：[`export_v22_answeronly.py`](../../pipeline/sft/export_v22_answeronly.py) 从库 A 对齐的 3,983 题导出，保留原首轮用户消息与原图，目标改为 `<answer>X</answer>`；GRPO 题库同题，只替换 system。
- SFT：[`sft_v22_answeronly.yaml`](../../pipeline/sft/sft_v22_answeronly.yaml) 与库 A 的 SFT 超参相同（有效 batch 48、2 epoch、lr 1e-4、LoRA r16），只换数据集；合并 [`merge_answeronly.yaml`](../../pipeline/sft/merge_answeronly.yaml)。GRPO：[`grpo_ao.sh`](../../pipeline/grpo/scripts/grpo_ao.sh)，原版 GRPO-A 配方，记录 rollout。
- 排队（服务器自动接续）：[`chain_gpu01.sh`](../../pipeline/grpo/scripts/chain_gpu01.sh) 卡 A 结束 → 答案式 SFT（约 1 小时）→ 合并 → KL GRPO；[`chain_gpu23.sh`](../../pipeline/grpo/scripts/chain_gpu23.sh) 卡 B 结束且 SFT 就绪 → 答案式 GRPO → 无 KL 对照重跑。
- 风险：答案式 SFT 从未学过裁图，GRPO 里可能一直直答。

## 5. 之后的报告格式

每组实验按论文表格记录，分下游 benchmark 给准确率，并附开销列：实际训练（监督）token 数、优化器更新次数、rollout 开销（rollout 条数、生成 token、GPU 小时、墙钟时间）。

## 6. 结果表（论文格式，持续补全）

### 表 1：E1 分 benchmark 准确率

E1 = 552 题 × 4 次采样，T=0.7，同种子；按训练规则判分（`ok`）。服务器上的评测题只有四个来源：MME-RealWorld 339 题、MMMU-Pro 39 题、V* 112 题、WorldBench 62 题；没有 HR-Bench、TreeBench、VisualProbe。脚本 [`bench_table.py`](../../pipeline/grpo/v51_cc/bench_table.py)。

| 方法 | MME-RW | MMMU-Pro | V* | WorldBench | 总体 | 每题裁图 | 直答率 | 每样本生成 token |
|---|---|---|---|---|---|---|---|---|
| SFT cold-start w/o RL（库 A） | .552 | .263 | .797 | .290 | .552 | 1.01 | .43 | 161 |
| 答案式 SFT w/o RL | 待测 | | | | | | | |
| 答案式 SFT + GRPO | 待测 | | | | | | | |
| SFT + GRPO（原版 GRPO-A，@160） | **.700** | .288 | **.866** | .339 | **.664** | 2.31 | .00 | 276 |
| SFT + GRPO + KL | 待测 | | | | | | | |
| crop-floor λ=0.025（@160） | .681 | .301 | **.866** | .298 | .649 | 1.38 | .04 | 189 |
| crop-floor λ=0.05（@160） | .672 | .340 | .846 | .343 | .647 | 1.23 | .03 | 190 |
| crop-cost λ=0.01（@40，塌缩） | .507 | .308 | .681 | .355 | .511 | 0.08 | .95 | 69 |
| 卡 A（价格通道） | 待测 | | | | | | | |
| 卡 B（价格通道 + 只丢全错组 DS） | 待测 | | | | | | | |
| 参考：库 B SFT / SFT + GRPO | .540 / .651 | .301 / .340 | .732 / .862 | .335 / .351 | .539 / .638 | 1.59 / 1.77 | .41 / .00 | 297 / 329 |

加粗只标 MME-RW、V* 与总体的最好值；MMMU-Pro（39 题）与 WorldBench（62 题）题数少，单格差异 3–5 个点在噪声内，不加粗。

### 表 2：训练开销

GRPO 均为 2 张 RTX Pro 6000，每步 12 题 × 16 条 rollout（192 条），一次优化器更新。生成 token = 策略实际采样、参与损失的 token（由 `timing_s/gen` 与每 token 耗时换算，与 rollout 日志逐条求和一致）；处理 token = 进入 actor 的全部 token（含提示、回执与图像）。脚本 [`cost_table.py`](../../pipeline/grpo/v51_cc/cost_table.py)。

| 方法 | SFT 更新次数 | GRPO 更新次数 | rollout 条数 | 生成 token | 处理 token | GRPO 步时合计 | GRPO GPU 小时 |
|---|---|---|---|---|---|---|---|
| SFT（库 A） | 284（有效 batch 48，3 卡 4,240 s） | — | — | — | — | — | — |
| 原版 GRPO-A | 284 | 165 | 31,680 | 7.47M | 97.4M | 8.32 h | 16.6 |
| crop-floor λ=0.025 | 284 | 165 | 31,680 | 6.09M | 87.4M | 6.79 h | 13.6 |
| crop-floor λ=0.05 | 284 | 165 | 31,680 | 5.80M | 84.3M | 6.56 h | 13.1 |
| crop-cost λ=0.01（停于 56 步） | 284 | 56 | 10,752 | 1.03M | 23.0M | 1.59 h | 3.2 |
| 卡 A / 卡 B / KL / 答案式 | 训练中或排队 | | | | | | |

SFT 当时没有记录训练 token 数（`num_input_tokens_seen=0`），之后离线统计补上；答案式 SFT 已打开该开关。卡 B 的 DS 会额外生成被丢弃的批次，届时 rollout 条数按实际生成计。
