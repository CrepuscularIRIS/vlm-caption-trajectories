# 三轮失败后的 Pro 重设计任务

状态：请求算法审查与一个最小可证伪设计；不授权训练。整理日期：2026-10-22（America/New_York；执行者报告使用 10-23 CST）。

## 任务与阅读顺序

请重新检查本对话上一版设计，不预设“换成不饱和损失”已经找到原因。用户希望复用已经提取的 idea card / pattern，寻找适合当前失败的建模方式。目标是模型自主按状态决定答题／继续观察，在保留准确率和必要 recovery 的情况下减少无效裁图；直答率不是目标。

结果代码固定于 `ba449ba`；随后 `6c2bddf88d12e7b4218a317f37fe472393504ad5` 只增加报告中的下一步建议等文档。请报告实际读取了哪些文件、哪些仍未读，不把仓库自评当成独立验证。

1. [报告 §7 和新建议 §7.5](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/6c2bddf88d12e7b4218a317f37fe472393504ad5/docs/30_rl_credit_report_20261022.md)。
2. [DL 的 PREREG](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449ba/pipeline/grpo/v52_dl/PREREG.md)、[生产 loss](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449ba/pipeline/grpo/v52_dl/decision_loss.py)、[adapter](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449ba/pipeline/grpo/v52_dl/easyr1_caption.py)、[协议](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449ba/pipeline/grpo/v52_dl/caption_v52i.py)。
3. [audit](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449ba/pipeline/grpo/v52_dl/dl_audit.py)、[compare](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449ba/pipeline/grpo/v52_dl/dl_compare.py)、[64 状态及结果](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449ba/logs/dl/compare_dl.json)。
4. [四格实现](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/tree/ba449ba/pipeline/grpo/v51_fc)、[零信用实现](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/tree/ba449ba/pipeline/grpo/v51_fcz)，沿报告读日志即可，不必重新通读历史全部实验。
5. [上一版 Pro 完整设计](../decision_loss_20261022/PRO_RESPONSE.md)、[导师视角与概念区分](../decision_loss_20261022/MENTOR_INTERPRETATION.md)、[已有完整卡库和模式索引](../decision_loss_20261022/README.md)。GitHub HTML 失败可取同版本 raw 文件；不能访问就明确缺口。

## 已有观测与解释边界

| 实验 | 报告中的观测 | 不能由此直接推出 |
|---|---|---|
| fc 对 r1 | E1 准确率差 +.003 [−.018,+.023]；裁图 +.54 [+.43,+.63] | 不能称准确率已证明不变；不能否定所有信用分配 |
| zero 对 r1 | 准确率 −.025 [−.045,−.005]；裁图 −.31 | 它确实少裁，但没有实现保准确率的目标；fc 对 zero 的优势不等于解决问题 |
| DL 对匹配 β0 | U_choice −.0254 [−.0341,−.0179]；裁图 −.4375；直答 +.0703；clean 准确率 −.0547 [−.1484,+.0391]，E2 −.0234 [−.1094,+.0703] | 宽区间不证明非劣；固定参考策略 Q 下的 U_choice 不等于新策略完整收益 |

DL：r1@40 开始，v5.2i（WORD 仅为 intent，不锁后续动作），K=4+4，20 次更新，fresh AdamW，lr=1e−6，无 warmup，β=1/256，H0=12；β0 控制同配方。训练共约 3.7583 GPU 小时。总预算验收还须包含 audit、eval、失败重跑；训练时间不能独自证明全流程低于 8 GPU 小时。

参考 audit 的 64 状态：D 正/负/零分别 5/46/13；61/64 的直答概率相对参考上升；平均 p 从 .0590 到 .1276。报告据此解释为“裁图侧饱和，少数直答状态推动所有状态”。这是待区分的机制解释。audit 的状态、D 和梯度不等于实际训练批次的分布。

执行偏离已记录：step-1 gate 没自动运行，到 step 8 才手动检查；事后通过不能抹去偏离。评测环境缺变量导致两次重跑，报告说权重未改。E1 多次用于设计，只是开发诊断；recovery 实际只有 14 状态、2 状态缺失。

## 这次首先要独立解决的问题

1. **局部梯度小是否真是缺陷？** 对 D<0 且 p≈0，BCE 已经选对，梯度小符合目标。为何少数状态的修正会改变其他状态？区分参数共享造成的干扰、全局动作偏置、训练 D 噪声、优化器/梯度尺度及 RLOO 交互。不把 held-out p 的变化当作训练因果证据。
2. **有限 K 估值与 weighted BCE。** 请复核你上一版关于带噪 D 的推导：若 δ=E[D_hat]、η=E|D_hat|，期望 logit 梯度为 η(p−1/2)−δ/2。它是否足以解释本次结果？缺什么证据？K=4 对 .01 成本差的可识别性如何？已有分叉如何进行独立拆分、重采样或负控？不要根据同一 audit 调参再宣称通过。
3. **候选修复必须有数学边界。** 直接 −D·z 无下界；绝对 logit 回归 z≈D/τ 可能抹去原置信度；对称符号加权可能改变状态分布与目标。比较这些与参考策略相对 log-odds、有限步策略改进、约束更新的实质差别。不能仅凭“非饱和”就推荐再训。
4. **接口核查。** 实际打分是 `<answer` 与 `<grounding` 的多 token 前缀，非闭合尖括号完整标签；检查其与自然解码动作事件、温度、合法格式的对应。caption 位于动作之前，但 WORD 更早；分叉与自然策略是否同一支持集？数学正确不代替这些绑定。
5. **全局偏置必须成为强简单对照。** 如何区分学会按状态决策与给所有状态加同一个直答偏置？给预算匹配、不会使用评估标签偷调参的控制。必要时仅用已有结果拟合描述性常数偏移；要明确描述性复算不是新实验的因果证据。
6. **势能路线。** 若考虑势能塑形，写清 terminal boundary / 折扣 / return-to-go / token 聚合；它与 fc 的系数替换为何不同？保持最优策略并不承诺在有限预算里减少裁图。不要仅换名字重跑失败配方。

## 卡与模式：从匹配出发，不以卡数证明价值

复用完整的 58 张机制卡、14 个 parent pattern、19 个 subpattern。下面只作检索入口；请实际读取并收敛到最多 3 张主卡，可拒绝不适合的卡。

| 入口 | 这次的具体用途 | 匹配边界 |
|---|---|---|
| [K11 工具收益条件奖励](../decision_loss_20261022/mechanism_cards/K11-benefit-conditioned-tool-reward.md) | 核查动作相对收益及已有失败边界 | 奖励与辅助损失不自动等价；不是首次想到 |
| [K19 MC 步标签](../decision_loss_20261022/mechanism_cards/K19-monte-carlo-step-labels.md) | 估值对象、分叉噪声、监督转化 | 局部正确率不是完整续跑 Q |
| [K36 最小差异偏好对](../decision_loss_20261022/mechanism_cards/K36-minimal-difference-preference-pairs.md) | 比较对象固定到同一真实状态 | 只有局部差异不保证监督正确或无噪声 |
| [K27 限制更新范围](../decision_loss_20261022/mechanism_cards/K27-restrict-update-scope.md) | 检查修正路由为何伤害其他状态/能力 | 小 lr 已用过；LoRA 不天然保证无干扰 |
| [K29 保存 logit 一致性](../decision_loss_20261022/mechanism_cards/K29-stored-logit-consistency-replay.md) | 参考置信度保护、概率与 logit 约束的差别 | 小规模分类到生成模型仅为跨域类比 |
| [K57 分离回答权限](../decision_loss_20261022/mechanism_cards/K57-decouple-answer-authority.md) | 审查“路由能力”和“答题能力”是否需要分开建模 | 冻结外部 inspector 会改变自主性与推理成本，不能默认采用 |

对应模式：[diagnose_then_target](../decision_loss_20261022/patterns/parent/diagnose_then_target.md)、[repair_degenerate_signal](../decision_loss_20261022/patterns/parent/repair_degenerate_signal.md)、[constrain_what_changes](../decision_loss_20261022/patterns/parent/constrain_what_changes.md)、[measure_example_utility](../decision_loss_20261022/patterns/parent/measure_example_utility.md)。

每个采用的卡请写：对象/可见信息 → 比较操作 → 信号 → 梯度接收者 → 预期变化 → 最小反证。卡中引用是检索线索；正式借鉴/新颖性判断请核对关键原文，列实际读到的版本。经典 cost-sensitive classification / policy iteration / conservative policy improvement 等也应比较，避免只在 VLM 卡里寻找重新命名的标准算法。

## 要求的交付与停止点

中文，先人话解释，再给可核验算法。

1. 明确哪些失败解释已支持、哪些仍是假说，以及你上次设计的哪条假设被挑战。
2. 一张紧凑的卡匹配表（最多 3 张主卡，区分直接机制、部分类比、缺口），有根据的最近方法对比；允许结论是标准方法适配，不强求新颖性。
3. **一个主方案 + 一个必要匹配控制**。精确定义状态、动作、回报、loss、reference、stopgrad、reduction、token mask、参数更新范围；推导 z 的梯度，给 D 正/负/零和极端置信度的手算例子。必须解释如何防止统一偏移、噪声追逐、无界 logit，以及保护答题和 recovery。
4. **最小分阶段 Probe**：优先零 GPU 的现有数据/数学检查，若不足才列一次小 GPU 检查或短训的必要条件。明确有区分力的判据、停止条件和全流程预算。不要大矩阵、不把通过格式审查当科学通过。任何新训练仅是提案，等待用户决定。
5. 给导师看的 200–300 字解释：这次真正改的对象是什么，为什么它可能解决失败，哪些结果会让我们停止。

此次 Codex 只组织公开证据并收集你的设计，不修改 CC 管理的 Open Scientist study，不执行新训练。
