# 代码、训练日志与轨迹：给导师 check 的索引（2026-10-18）

按导师要求"把一些 log、代码都更新到 github repo 上"，本次一次性补齐：

| 内容 | 位置 | 说明 |
|---|---|---|
| 全流程代码 | [`pipeline/`](../../pipeline/README.md) | 协议 → SFT 数据与 LLaMA-Factory 配置 → GRPO（EasyR1 602a128 + 多轮裁图插件，四个变体）→ E1/E2 评测与分析脚本；附 EasyR1 本地改动 diff。只做了路径脱敏（[SCRUB_NOTES](../../pipeline/SCRUB_NOTES.md)） |
| 构造、审核与筛选代码 | [`pipeline/construct/`](../../pipeline/construct/README.zh.md) | 题池 → 协议 → 多教师真实裁图生成 → 审核 → 筛选 → 库 A 装配（V2 / V2.2）；库 B 插值构造与交付。79 个文件，按实际运行版本收录，只做路径替换。附已知问题；**裁图部分可能重新构建，方案已有** |
| 训练日志 | [`logs/`](../../logs/README.md) | 7 个 GRPO run 的逐步指标 CSV 与完整解析配置；2 个 SFT run 的 loss 曲线；E1/E2 汇总文本 |
| 轨迹 | [`trajectories/`](../../trajectories/README.md) | 库 A（V2.2，4,040 条）与库 B（v4c 插值，3,983 条）的全部 assistant 轮；**已去掉题干、选项和图像**，按 `bench + item_id` 可从公开基准取回；附校验脚本与统计 |

## 读之前需要知道的三点

1. **所有 run 都没开 DAPO 的 dynamic sampling**（`online_filtering=false`），全对/全错组都留在批内。但这件事和裁图代价相互牵连：不开时，裁图次数只在全对组里拉开奖励差，而 std 归一化会把这 0.01 量级的差放大成满额优势（第一轮直答塌缩的原因）；按 accuracy 打开后，全对组被整组丢掉，λ=0.01 只剩约 1% 的信号。怎么设置，与训练优化一起讨论后再跑。
2. **clip-higher 配了但从未起作用**：每个 rollout batch 只做一次完全 on-policy 的更新（`global_batch_size = rollout_batch_size = 12`、`ppo_epochs=1`），重要性比恒为 1，所有 run 的 `pg_clipfrac` 每步都是 0。要让它生效需要把一批拆成多次更新，待讨论。
3. **第二轮的 std 下限（改 GRPO 分母）是我们自己的改动，已停用**；原因与过程见[阶段报告](24_stage_report_20261018.md)和下一次回复。

## 更正

- Adaptive-CoF 不再作为参考依据（[docs/22](22_dual_library_eval_20261016.md) 文首已加更正）。之后引用论文先读原文、写筛选记录。
