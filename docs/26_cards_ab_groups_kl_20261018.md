# 卡 A / 卡 B 中期、rollout 分组统计、KL 与新增基线（2026-10-18 晚）

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

训练 rollout 第 34–40 步：卡 A 每题裁图 1.74、直答 7%、通过验证的裁图 19%；crop-floor λ=0.025 为 1.64 / 7%。裁图数在所有 run 中都随训练上升，卡 A 尚未压低裁图。逐步指标：[`logs/grpo/grpo_v3_A_cc_metrics.csv`](../logs/grpo/grpo_v3_A_cc_metrics.csv)、[`logs/grpo/grpo_v3_A_cc_dsasym_metrics.csv`](../logs/grpo/grpo_v3_A_cc_dsasym_metrics.csv)（训练中导出，非最终）。

## 2. rollout 按组划分：全对 / 全错 / 有对有错

导师要求：把 rollout 按组分成全对、全错、有对有错三类，分别统计直答与裁图的正确率、advantage、监督 token 数。

- 脚本：[`pipeline/grpo/v51_cc/group_stats.py`](../pipeline/grpo/v51_cc/group_stats.py)，在服务器上对训练 rollout 日志计算，advantage 按各 run 实际公式由记录的分数重算。
- 全部五个 run 的完整表：[`logs/grpo/group_stats_20261018.md`](../logs/grpo/group_stats_20261018.md)。
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
- 计划（已排队，卡 A 跑满 165 步后在同两张卡上自动启动）：[`grpo_kl.sh`](../pipeline/grpo/scripts/grpo_kl.sh) = 原版 GRPO-A 奖励（只看对错）+ `low_var_kl`，`kl_coef=1e-2`（EasyR1 示例默认），记录 rollout。
- 对照重跑（卡 B 跑满后自动启动）：原版 GRPO-A 配置、同种子、记录 rollout，用于补齐对照组的分组统计，并与 KL 组构成同代码的有/无 KL 对比。接续脚本：[`wait_then_kl.sh`](../pipeline/grpo/scripts/wait_then_kl.sh)、[`wait_then_ctrl.sh`](../pipeline/grpo/scripts/wait_then_ctrl.sh)。
- 之后：按卡 A / 卡 B 的结果，把较好的一个与 KL 版 GRPO 组合。

## 4. 新增两个基线（导师要求）

| 基线 | 内容 | 状态 |
|---|---|---|
| SFT cold-start w/o RL | 库 A SFT 起点，不做 RL | 已有 E1 总体 .552；补分 benchmark 表 |
| 答案式 SFT cold-start w/ RL | 同题同图，SFT 目标只有最终答案（无 caption、无裁图），再做与原版 GRPO-A 相同的 GRPO | 正在核对导出脚本与 GRPO 环境提示的兼容性 |

## 5. 之后的报告格式

每组实验按论文表格记录，分下游 benchmark 给准确率，并附开销列：实际训练（监督）token 数、优化器更新次数、rollout 开销（rollout 条数、生成 token、GPU 小时、墙钟时间）。
