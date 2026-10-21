# 决策点基线：导师四种情况的信用分配（2026-10-22，进行中）

导师 10-21 的第二点是信用分配：按"用工具前对 / 错 → 用工具后对 / 错"的四种情况，改哪些 token 接收结果（原话见 [docs/30](30_sec6_and_rl_objective_20261021.md) 第 0 节）。

这次的方法用 Open Scientist idea run `tool-credit-assignment` 设计，方法库是 10-19 自建的 `vlm-post-training` v1，最后判定为 pursue。
- idea card：[`docs/idea_cards/tool_credit_assignment_20261022.md`](idea_cards/tool_credit_assignment_20261022.md)
- 撞车核对：[`docs/idea_cards/tool_credit_assignment_scoop_20261022.md`](idea_cards/tool_credit_assignment_scoop_20261022.md)

单种子，结果出来后补表。

## 1. 方法（一句话）

在第 2 次及以后的每个工具决策点 s_k，让**正在训练的策略自己**分叉 m 次，预填 `Step k | commit` 直接作答，并按标准答案打分，得到"现在就答"的价值 b(s_k)。

第 k 轮的**所有** token 都拿 r_i − b(s_k)，不管这一轮采到的是裁图还是作答；第 1 轮保持 RLOO 优势。
- 期望梯度和 RLOO 相同（满足假设 a–d 时），奖励里什么都不加，只改每个样本得到的信用。
- 四种情况自然落在后续裁图轮上：有效 ≈ +1，冗余 ≈ 0，误导 ≈ −1，失败探索 = −b(s_k)。

## 2. idea run 经过

| 步骤 | 结果 |
|---|---|
| 找瓶颈 | 只用最终答案奖励时，每一轮拿到的优势相同；四种情况需要"策略自己在调用前的状态"，rollout 里观测不到。冻结读者的状态不等于策略的状态（读者正确率 .53，策略 .66） |
| 第 1 版 | "现在就答"分叉值只用在裁图轮。一致性审查发现基线依赖被打分的动作，会多出一项暗含的停止倾向 −π(1−π)(V_stop − J)，小例子复现了这个偏差 → 审查**否决** |
| 第 2 版 | 同一个基线用在决策点的所有 token 上，和动作无关。一致性审查在 3 层决策的小例子上验证无偏（7 个状态都在 3 个标准误以内） |
| 撞车检查 | arXiv 接口限流，改用 web search 加抓取，共 46 篇。CounterCredit（2609.22910）部分重叠：同样的"现在就答"值，但当价格加进奖励 |
| 审查第 1 轮 | 判定修订：补和 CounterCredit / TACO / VinePPO / CARL / IGPO 的区别；PERM 改为同题同轮次内打乱；crop-only 臂放进第一次检验；加一组 2B 模型 |
| 审查第 2 轮 | 判定通过；新颖性 2、竞争 2、可行性 4，被抢先风险高；可能出现小效应或零效应 |
| 开训前补查 | 又核对了 14 篇，以及 161 篇引用 BRPO / MRT 的论文，没有发现相同机制。最接近的是 BRPO（2505.13438）、SHAPE（2604.06636）、MRT：都用同样的强制作答值，但都在单轮推理里，而且当奖励或塑形用 |

## 3. 第一次检验（Open Scientist study `rl-objective-credit`，活动 t-db-wave1）

| 步骤 | 内容 | 状态 |
|---|---|---|
| 1 单元测试 | 小例子直接调用生产代码 `apply_db`：DB 和 PERM 无偏，crop-only 的偏差符合理论公式 | ✅ 服务器上 4 项通过 |
| 2 重叠检查 | E1 与库 A、RL 训练集按 uid、题面和图像感知哈希比对 | 运行中 |
| 3 离线门槛 | r1 第 40 和第 160 步的 checkpoint，各取 200 题 × 16 条 rollout，每个后续决策点 4 个分叉。要求 DB/RLOO 信用二阶矩比的区间上界 < 1，且预填检查通过（字母一致率 ≥ .80，准确率差 ≤ .10） | 待跑 |
| 4 第一波 | DB（GPU 0,1）与 crop-only（GPU 2,3），r1 的配方：RLOO，λ0.01，165 步 | 门槛通过才跑 |
| 5 评测 | 第 80 / 160 步的 E1 / E2，对比 r1、r0 | — |

- **事先写好的预测：** 第 160 步 E2 上 DB 高于 r1（区间不含 0）；E1 准确率不降；DB 的裁图数落在 r1 的区间内，crop-only 落在区间外。
- **竞争解释：** DB 和 r1 一样，说明只改信用落点、不改目标没有作用，这正是导师说的"瓶颈在目标函数"；或者门槛不过，记为不可测。
- **冒烟测试：** 训练 1 步时，146 个分叉、73 个决策点、0 截断；后续轮次信用的二阶矩 0.209 → 0.186；单步 132 秒。

## 4. 代码

| 内容 | 路径 |
|---|---|
| 训练插件（分叉 + 优势改写，模式 db / perm / croponly / rloo） | `pipeline/grpo/v51_db/`（`db_credit.py`、`easyr1_caption.py`、`test_db_credit.py`） |
| 启动脚本 | `pipeline/grpo/scripts/grpo_db.sh` |
| 离线门槛、重叠检查、对比 | `pipeline/eval/db/`（`offline_forks.py`、`overlap_audit.py`、`compare_db.py`） |
