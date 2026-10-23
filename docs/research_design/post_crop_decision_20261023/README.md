# 下一轮 Pro 输入：第一次裁图后，何时已经足够？

**已完成 Pro 咨询；先看 [结论与下一步](READOUT.md)，另有 [完整回复](PRO_RESPONSE.md)、[实际提示词](SENT_PROMPT.md)和[完成回执](RESPONSE_RECEIPT.json)。**

以下保留发送时的问题包。状态：设计咨询；停止首轮 Stage 1 的决定保留。未经用户/导师新安排，不采样、不训练、不启动小头。先读 [工程与建模教训](LESSONS.md)，再读 [Stage 0](../../../logs/dl/stage0/README.md)。数据基线 `5399cba`。

补充：[后续奖励与势能的代数核对](ALGEBRA.md)。这是 Codex 自查，不改变已发送的输入提交。

## 要解决的一个问题

用户目标是 base model **自主**保住答题和 recover 能力，同时减少第一次之后的冗余裁图。当前事实只有“首轮空间太小”和“部分轨迹继续裁图”；**后续存在冗余且可学习，仍是假设**。不要将外置检查器/小头的成功直接当成 base model 学会了。

候选研究问题：给定真实首裁回执及历史，同一个参考模型现在作答与再取证后自然完成，差别是否足够大、足够稳定，并且能由当时可见的信息预测？失败可能来自不该再看、该看但取错、已有证据没用好、或协议把动作提前固定。先区分这些解释。

## 先匹配已有卡，而不是另列一堆方法

完整卡库：[mechanism_cards](../decision_loss_20261022/mechanism_cards)；[parent patterns](../decision_loss_20261022/patterns/parent)；[subpatterns](../decision_loss_20261022/patterns/sub)。以下只是迁移候选，不能当原文核验或创新证明：

| 卡 | 这次能借什么 | 不能照搬什么 | 最小否证 |
|---|---|---|---|
| [K19](../decision_loss_20261022/mechanism_cards/K19-monte-carlo-step-labels.md) | 同状态续写估计答案可达性 | 可达性不是视觉事实真伪，K4 可能分不出小成本差 | 合法配对后停止价值无空间或标签无法稳定判别 |
| [K11](../decision_loss_20261022/mechanism_cards/K11-benefit-conditioned-tool-reward.md) | 按工具增益而非统一次数奖励 | item 级“需要看”不等于每一后续轮都需要看；不默认提高 λ | 状态条件信号不优于简单预算/原分数规则 |
| [K57](../decision_loss_20261022/mechanism_cards/K57-decouple-answer-authority.md) | 显式检查证据是否足够，拆开停止与获取 | 外置 inspector 替代答题不符合最终自主 base 目标；自身会误判 | 候选监督器与真实停止收益不一致，或收益仅靠外部能力 |

主 pattern：[diagnose_then_target](../decision_loss_20261022/patterns/parent/diagnose_then_target.md)；辅助：[measure_example_utility](../decision_loss_20261022/patterns/parent/measure_example_utility.md)。请 Pro 可以否决上述匹配、改选已有卡，但只收敛为一张主 card。

## 请 Pro 交付什么

1. 用人话重述导师的建模顺序，并审查本次教训中是否有误判。不要将“BCE 饱和导致全局漂移”当已证实根因。
2. 给出**一张问题/机制卡**：对象、时点、可见信息、两路比较、学习信号、接收梯度的参数/token、预期行为、最小否证；先说明为什么这是后续停止/继续决策，和 advantage、credit assignment 各有什么关系。
3. 先提出一个最小、无训练的诊断。明确现有首轮 fork 为什么不能回答后续反事实；先用已有完整轨迹能核对什么，何时才确实需要新 fork。样本单位、选择偏差、合法共同前缀、固定策略、invalid、预算耗尽、长链必要探索、题级聚类、样本精度与停止规则都要说清。不得靠筛选最终错题或 caption 含答案制造阳性率。
4. **协议难点必须正面解决**：v5.2i 只松绑 Step 1。选择 receipt 后且下一轮 WORD 前，或 caption 后改协议，究竟测的是什么？若选 WORD 前，两路连 caption 都可能不同；若选后者，动作已被绑定。不能既声称完全同 caption，又不承认支持集变化。
5. 保留当前奖励 $Y(1-.01N)$ 作参照，同时分别看准确率、后续裁图数、生成/图像 token、真实推理成本和恢复能力。若用“剩余成本”，请推导它与既有成功门控奖励的差别：历史已花的裁图数乘着最终 Y，不一定是可随意删除的常数。
6. 诊断若通过，才给一条条件化算法建议，写清是否改目标、怎么防止全局少看却掉分、如何最终写回 base model。若只提训练外控制，明确距自主行为还差什么。势能 shaping 不能只给名称，需边界条件与是否保持目标的推导。
7. 与最强简单对照比较（自然继续、固定停止策略或 train-only 阈值，选择必要的最少项）。全轨迹收益不得把多次单点优势直接相加；说明部署状态分布变化。
8. 给出研究价值的**待证命题**、最近邻工作的原始出处、确实读到的范围，以及失败后停止还是转向证据利用。允许结论是“当前不值得训练”。先出可与导师讨论的设计，不给自动开跑命令。

## 参考入口

- [三轮失败实验总报告](../../30_rl_credit_report_20261022.md)
- [前轮 Pro 设计全文](../decision_loss_redesign_20261022/PRO_RESPONSE.md) 与 [读解/门槛](../decision_loss_redesign_20261022/READOUT.md)
- [导师意图整理](../decision_loss_20261022/MENTOR_INTERPRETATION.md)
- 决策点基线历史以总报告和前轮设计中的归档说明为准；旧 docs/31 路径当前不存在，不作为可读输入。

所有模型意见为咨询输入，不能替代实际图像审查、counterfactual rollout 或实验。Open Scientist 记录由 CC 管理，本轮不写共享 study。
