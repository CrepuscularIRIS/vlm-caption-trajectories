# Probe 阶段实验协议（请导师确认）2026-10-24

**定位：** 现在还是 Probe 阶段，方法没有定型。这一轮目标只有两个：
1. 按 10-23 的意见把 baseline 系列补齐；
2. 把 setting 修正后的标准 GRPO 跑到收敛附近。

模型只有 4B，所以每组用半个 Mini-o3 的规模、2 张卡，不追全量复现。

导师原话逐条见 `lark/mentor_words_20261024/导师原话_全量.md`（M01–M18），逐条对账见 [docs/38](38_mentor_requirements_audit_20261024.md)。

## 1. 共同 RL 配置（所有 RL 组相同，只改各组自己的开关）

| 项 | Mini-o3（论文 / repo） | 我们的 Probe 配置 | 说明 |
|---|---|---|---|
| 底座 | Qwen2.5-VL-7B | Qwen3-VL-4B（A-SFT ckpt 或原模型） | 模型小一半 |
| GPU | 64 张（8×8） | **2 张** RTX PRO 6000 96GB / 组 | 4 张卡同时跑 2 组；update micro 2、experience micro 4 |
| RL 数据 | VisualProbe 4k + DeepEyes 8k | 自有 8k 题池（已去掉 V*、E1 评测题及近重复图） | 未用他们的数据 |
| 每步 | 256 题 × 16 条 | **128 题 × 16 条** | 用步数换批大小 |
| 组筛选 | rejection sample：零方差组丢掉再补批 | 不筛选（`online_filtering=false`） | **不同** |
| mini-batch / 每步更新次数 | 32 / 8 | 32 / **4** | 更新是真正 off-policy 的，clip 生效 |
| 步数 | 曲线约 200 步 | **200 步**（约 3.2 个 epoch，800 次更新） | 题目曝光量约为 Mini-o3 的一半 |
| lr / KL / entropy | 1e-6 / 无 / 无 | 1e-6 / 无 / 无 | 同 Mini-o3；KL 见 §4 Q1 |
| clip low / high | .2 / .3 | .2 / .3 | 同 |
| max_grad_norm | 1.0（verl 默认） | 1.0 | 旧配置 0.1 每步都被裁 |
| 训练时轮数上限 | 6 个生成轮（最多成功裁图 5 次） | 6 裁上限，但第 7 裁执行后才截断，且同时作用于验证（docs/40 I7，待修） | **不同** |
| 超轮 / 超长轨迹 | over-turn masking：A·M | 同样置零，不贡献梯度 | 已实现（`CAPTION_MASK_TERMINALS`） |
| 每轮生成上限 | 8,192 token | 640 token | **不同** |
| 图像像素 / 总预算 | 2M px / 32K | 589k px / 4k prompt + 7k response（含多轮回执与视觉 token） | **不同** |
| loss 聚合 | 每个 micro 内 token 平均，micro 等权 | 按全 mini-batch 有效 token 加权 | 不同（变长 / 掩码轨迹下权重不一样） |
| 奖励 | LLM judge | 选择题精确匹配 | 不给格式分，不给标签分 |
| 评测 | 多次采样 | 正式表用 greedy + 分 benchmark；E1 继续报告 T=.7×4 | 见 §3 |

**更正（docs/40 I11）：** 这不是 Mini-o3 的等价复现或半规模复现，只是借鉴了它的一部分参数：每步多次更新、clip .2/.3、无 KL、grad norm 1.0、超轮轨迹做 mask。

组筛选、每轮 token 预算、轮数语义、loss 聚合、分辨率、奖励都和 Mini-o3 不同。Mini-o3-7B 作为外部模型，按它自己的原生协议单独评测。

## 2. 实验组

| ID | 条件 | 起点 | 需训练 | 状态 |
|---|---|---|---|---|
| B0 | Base direct（无工具 prompt） | 原模型 | 否 | E1 .490 |
| B1 | Base + 原生 zoom 工具 | 原模型 | 否 | E1 .554 |
| B2 | Base zero-shot + caption 协议 | 原模型 | 否 | E1 .033（93% 不符格式） |
| B3 | Base + SFT | A-SFT | 否 | E1 .552 |
| **B4** | Base + SFT + GRPO（标准 GRPO） | A-SFT | 是 | **2 卡运行中** |
| B5 | Base + GRPO（无 SFT） | 原模型 | 是 | 待确认工具格式（§4 Q2） |
| B6 | Base + GRPO without tool | 原模型，direct prompt | 是 | 待排 |
| B7 / B8 | AdaptVision / CounterCredit | 同底座适配 | 是 | 待适配 |
| D1 | DAPO（DS + clip-higher）+ crop 惩罚 | A-SFT | 是 | 10-18 欠账 |
| D2 | Dr.GRPO（去 std）+ L_budget | A-SFT | 是 | 10-18 欠账 |
| D3 | 首动作监督（完整 run） | A-SFT | 是 | 10-22 欠账 |
| X1 / X2 | Mini-o3-7B、DeepEyes-7B | 官方权重 | 否 | 下载中 |

RL 组共 9 个（B4–B8，D1–D3）。每批 2 组同时跑，单 seed。

## 3. 报告口径（每组相同）

- **一张 trade-off 图**：
  - 横轴为成本：每题首裁、每题后续裁图（分开画）、每题处理的总 token；
  - 纵轴为准确率。
- **表格分下游 benchmark**：V*、HR-Bench 4K/8K、MME-RealWorld（干净子集）、VisualProbe。
- **训练账单**：每组报告训练 token、更新次数、rollout 条数和 GPU 时。

## 4. 请导师确认的问题

1. **KL**：B4–D3 默认关闭 KL，和 Mini-o3 一致。需要加 KL 吗（例如 0.01）？
2. **B5（无 SFT 的 GRPO）的工具格式**：原模型在 caption 协议下 93% 不符合格式，奖励几乎全为 0。
   - 我们建议 B5 用原模型自带的 zoom 工具格式（B1 的格式），
   - 否则这一组只能得到“学不会格式”这一个结论。
3. **规模**：Probe 阶段每组 128×16×200 步、2 卡，这个规模可以吗？等 B4 收敛后再决定是否放大到完整的 256×16。
4. **顺序**：先跑 B4+B6，再跑 D1+D2，再跑 B5+D3，最后跑 B7/B8。这个顺序可以吗？

## 5. 时间

按 B4 前几步的实测耗时填写（待填）。
