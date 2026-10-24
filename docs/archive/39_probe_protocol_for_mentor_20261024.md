> **已过时（2026-10-24 归档）：** 本页是旧分析和旧计划，结论与排期均已失效。导师要求现以内部全量对账为准（原话逐行、逐条对照、配置与任务缺口），已知代码问题见 [docs/40](../02_known_issues.md)。

# 按导师 10-23 意见的实验部署与参数（2026-10-24）

## 1. 导师原话 → 做法

| 导师 10-23 | 做法 |
|---|---|
| 时刻检查 setting | 先修完代码审查发现的问题（[docs/40](../02_known_issues.md)），再开训练 |
| 补 Qwen3-VL-4B 原模型 baseline 系列：不带工具、原生工具、caption 协议 zero-shot | B0 / B1 / B2，用修复后的同一套 runner 和预算重跑 |
| RL 规模拉上去；step 太少，没收敛 | 每步 12 题 → 128 题，更新 165 次 → 800 次（§3） |
| `max_grad_norm=0.1` 每步都被 clip | 改为 1.0 |
| 不要 multi-seed | 单 seed |
| baseline：Base direct、原生工具、caption zero-shot、Base+SFT、Base+GRPO（不做 SFT）、Base+GRPO without tool、AdaptVision/CounterCredit、两个外部模型 | B0–B8、X1/X2（§2） |
| 一张 trade-off 图：首裁、后续裁图分开，每题总 token | 所有条件用同一个评分脚本；训练中每条轨迹都记录 token |

其余设置**与原版 GRPO 相同，不额外改动**：只用正确性奖励、标准 GRPO、无 KL、不开动态采样、caption 协议不变。

## 2. 卡的部署（4 张 RTX PRO 6000 96GB，每组 2 张）

| 时间 | GPU 0,1 | GPU 2,3 |
|---|---|---|
| 第 1 批（进行中） | **B4** Base + SFT + GRPO，约 3.1 天 | **B6** Base + GRPO without tool，约 1.7 天 |
| B6 结束后 | B4 继续 | **B5** Base + GRPO（不做 SFT），脚本与 B4 相同，只把起点换成原模型 |
| 之后 | B0–B3 重测、外部模型 X1/X2 评测（只推理，每组几个小时） | B7/B8（AdaptVision / CounterCredit，同底座适配） |

每组 2 张卡的显存实测：峰值约 88GB/卡（vLLM 占 45%，FSDP 训练状态 + 激活占其余）。

**每步耗时实测**（冒烟 step 1，不含保存 checkpoint）：

| 阶段 | B4（带裁图，多轮） | B6（无工具，单轮） |
|---|---:|---:|
| rollout 生成 | 509 s | 286 s |
| 旧策略 log-prob | 181 s | 104 s |
| 策略更新（4 次） | 616 s | 345 s |
| **每步** | **约 22 分钟** | **约 12 分钟** |

## 3. 参数

| 参数 | 旧配置（10-16 → 10-22 各组） | 现在 |
|---|---|---|
| 起点 | A-SFT | B4：A-SFT；B5 / B6：原模型 Qwen3-VL-4B-Instruct |
| RL 题池 | 2,000 题 | 8,000 题（去掉了 V* 和 E1 评测题） |
| 每步 rollout | 12 题 × 16 条 | **128 题 × 16 条 = 2,048 条** |
| actor mini-batch | 12 题（每步 1 次更新） | **32 题（每步 4 次更新）** |
| 步数 / 更新次数 | 165 / 165 | **200 / 800** |
| 题目曝光 | 1,980 | **25,600**（约 3.2 个 epoch） |
| max_grad_norm | 0.1 | **1.0** |
| lr / warmup / 调度 | 1e-6 / 5% / constant | 同 |
| KL | 关 | 关 |
| clip low / high | .2 / .3（每步只更新 1 次，实际不生效） | .2 / .3（每步 4 次更新，**生效**） |
| 奖励 | 正确 = 1，否则 0 | 同 |
| 采样 | T = .7，top_p = 1，n = 16 | 同 |
| 序列预算 | prompt 4,096 / response 7,168；caption 每轮 640 token；最多裁图 8 次 | 同（B6 无工具，response 4,096） |
| 图像 | max_pixels 589,824 | 同 |
| 每卡 micro batch | update 2 / experience 2 | update 2 / experience 4 |
| checkpoint | 完整状态 | 同（完整状态可续训；每 50 步存一次，保留最近 2 个） |
| seed | 42 | 42（单 seed） |

**冒烟 step 1 的检查结果：**
- B6：grad_norm 0.87，不再被裁；clip 触发比例 0.2%，ratio clip 已经起作用；奖励均值 .534。
- B4：rollout 准确率 .549，与 A-SFT 的 E1 .552 一致；2,048 条中有 12 条非法、3 条超长。

## 4. 源代码

训练（[`pipeline/grpo/v51fix/`](../../pipeline/grpo/v51fix/)）在原版 [`pipeline/grpo/v51/`](../../pipeline/grpo/v51/) 上只改了三处：拒绝越界 bbox、token 账本、完整 checkpoint。

| 文件 | 作用 |
|---|---|
| [`grpo/scripts/grpo_v51fix.sh`](../../pipeline/grpo/scripts/grpo_v51fix.sh) | B4 / B5 启动脚本，上表参数全部在这里 |
| [`grpo/scripts/grpo_v51fix_notool.sh`](../../pipeline/grpo/scripts/grpo_v51fix_notool.sh) | B6 启动脚本（EasyR1 原生单轮） |
| [`grpo/v51fix/easyr1_caption.py`](../../pipeline/grpo/v51fix/easyr1_caption.py) | 多轮裁图 rollout |
| [`grpo/v51fix/caption_env.py`](../../pipeline/grpo/v51fix/caption_env.py) | 裁图环境（越界 bbox 在这里拒绝） |
| [`grpo/v51fix/reward_caption.py`](../../pipeline/grpo/v51fix/reward_caption.py) / [`reward_direct.py`](../../pipeline/grpo/v51fix/reward_direct.py) | 正确性奖励（caption / 无工具） |
| [`grpo/v51fix/training_ledger.py`](../../pipeline/grpo/v51fix/training_ledger.py) | 每条训练轨迹的裁图数、生成 token、处理 token |
| [`grpo/easyr1_602a128_local.diff`](../../pipeline/grpo/easyr1_602a128_local.diff) | 服务器上 EasyR1 的本地改动（与服务器一致） |
| [`grpo/scripts/build_direct_pool.py`](../../pipeline/grpo/scripts/build_direct_pool.py) | B6 数据：同一批题目与图像，prompt 换成 direct |
| [`eval/baselines/eval_base.py`](../../pipeline/eval/baselines/eval_base.py) | B0 / B1 评测（direct / 原生 zoom 工具） |
| [`eval/baselines/eval_caption.py`](../../pipeline/eval/baselines/eval_caption.py) | B2 / B3 / RL 模型评测（caption 协议） |
| [`eval/baselines/summarize.py`](../../pipeline/eval/baselines/summarize.py) | 统一评分与成本：准确率、首裁、后续裁图、每题总 token |

## 5. 状态

- B4 / B6 冒烟中，step 1 正常。冒烟与续训检查通过后，两组自动开始 200 步。
- 旧的 B0 / B1 / B2 分数作废（题面被截断、工具调用只执行第一个、越界框被执行，见 docs/40），会用修复后的代码重跑。
