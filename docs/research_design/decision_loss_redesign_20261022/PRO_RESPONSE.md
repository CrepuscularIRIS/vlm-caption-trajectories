# Pro 重设计答复（2026-10-23 收回）

- 状态：网页 Pro 显示 `Response complete`，已完整收回；这是模型咨询设计，不是实验结果或新颖性认证。
- 输入任务版本：`5c63a2803f3a4fe2f5105e461aafc780aa03e031`。
- 本地原始复制文本 SHA-256：`91c138fef0cf683e6e15fc683ab3b32aef397f19b3bba0ab38314f3383d6fb4c`。
- 以下保留答复正文，只将数学分隔符转换为 GitHub Markdown；文献阅读范围为 Pro 自报，Codex 未逐篇重新核验。
- 先看 [短版解释与 Codex 复核](READOUT.md)；[发送的提示词](SENT_PROMPT.md)。

---

## 先用人话说结论

**我不建议把“正确且自信的裁图状态上，BCE 梯度太小”当作已查明的故障，更不建议据此立即启动第四轮全参数训练。**

如果某个状态确实应该裁图，而模型也几乎必然裁图，它的监督误差已经很小，BCE 梯度小是正常的。真正需要解释的是：**为什么修正其他状态时，这些本来选对的状态也一起转向了直答？** 这里至少有三层问题：

- **全局直答偏移**描述模型输出怎样变化；
- **共享参数干扰**描述一个状态的更新怎样影响其他状态及答题能力；
- **有限 $K$ 噪声**描述监督信号怎样产生、经过 BCE 后又怎样变形。

它们可以同时存在，不能当作三个互斥原因。现有证据能确认广泛的直答偏移，能排除“全部变化只是一个常数偏置”，但**不能独立确认共享参数或 $K=4$ 噪声谁是主因**。

我建议的唯一主方案是：**冻结现有模型及其后续续跑，只学习一个很小的“直答与继续裁图的收益差”预测器，在预定的小概率下接管首个动作。** 唯一匹配对照是：用完全相同的分叉数据，充分校准原模型动作分数的一个全局阈值。

这样检验的是一个明确问题：

> 在不改变答题和后续恢复参数的情况下，隐藏状态中的信息，能否比原有动作分数的一维校准，更好地区分什么时候值得停止？

这是标准策略改进思想的受限应用，不应包装成新的工具信用理论。下面给出证据、推导及停止条件。**本轮仅阅读资料与做 CPU 统计复算，没有执行训练。**

---

# 一、独立诊断：哪些已经成立，哪些还没有

## 1.1 本次采用的实际版本

我按新入口读取，并区分了代码与后续报告的版本：

| 材料 | 实际版本及位置 |
|---|---|
| 新任务包 | `5c63a2803f3a4fe2f5105e461aafc780aa03e031`，`docs/research_design/decision_loss_redesign_20261022/README.md` |
| 真实实现、预注册、结果 | `ba449ba` 实际解析为 **`ba449babb4072a067f63b46419a169903d0cf407`** |
| 实验报告及下一步解释 | `6c2bddf88d12e7b4218a317f37fe472393504ad5`，`docs/30_rl_credit_report_20261022.md` 的 **§7–§7.5** |
| 上一版设计、导师解释、卡与模式 | 新任务提交下的 `docs/research_design/decision_loss_20261022/` |

以下判断以真实 loss、adapter 和逐状态结果为依据，没有将报告中的“通过”“原因已查明”作为证明。[新任务入口](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/5c63a2803f3a4fe2f5105e461aafc780aa03e031/docs/research_design/decision_loss_redesign_20261022/README.md)；[实验报告](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/6c2bddf88d12e7b4218a317f37fe472393504ad5/docs/30_rl_credit_report_20261022.md)

## 1.2 三个实验失败的方式不同

下面的差值均相对各自对照；fc/zero 与 DL 的评测集合不同，不能据此横向排列算法优劣。

| 实验 | 准确率差及 95% 区间，单位：百分点 | 平均裁图数差 | 可以支持的判断 |
|---|---:|---:|---|
| fc 对 r1，E1 | $+0.317\;[-1.766,+2.310]$ | $+0.536$ | 没有实现少裁图；准确率“不变”未获证明 |
| zero 对 r1，E1 | $-2.491\;[-4.484,-0.453]$ | $-0.305$ | 确实少裁图，同时损失准确率 |
| DL 对匹配 $\beta=0$，clean | $-5.469\;[-14.844,+3.906]$ | $-0.438$ | 少裁图，但宽区间不支持准确率非劣 |

DL 的共同状态指标还给出了更明确的局部否定证据：

$$
U_{\mathrm{choice}}
=\frac1{64}\sum_h
(p_{\mathrm{main}}-p_{\mathrm{ctrl}})\widehat D_h
=-0.02541363,
$$

$$
95\%\mathrm{CI}=[-0.03412416,-0.01794341].
$$

公开逐状态数据可以复现这个负均值；按原比较程序进行题目 bootstrap，也与公开区间一致。**旧预注册的否定判定应保留。** 但这个指标使用共同参考策略的 $Q$，不能直接解释成新模型整条策略的收益变化。[fc 结果](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/logs/fc/compare_fc.json)；[zero 结果](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/logs/fcz/compare_fcz.json)；[DL 逐状态结果](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/logs/dl/compare_dl.json)

两处报告措辞需要修正：

1. zero 已经减少裁图，失败在于没有保住准确率，不能写成“两个改写都没有减少裁图”。
2. 公开表格中，直答概率实际上是 **64/64 上升**。三个 `p_main` 已舍入为 1，只有 **61 行能反算有限 logit**；61 不是上升状态数。

另外，fc 优于 zero 还不能证明前后价值差包含了有用的状态信息。fc 的平均 crop credit 与固定 $-.01$ 的 zero 不同，**整体激励强度也变了**。缺少同等平均激励的简单对照，就不能把差异全部归给状态辨别。

## 1.3 “小梯度导致失败”的论证不完整

真实实现是：

$$
\ell_{\mathrm{BCE}}
=\widehat D_+\operatorname{softplus}(-z)
+(-\widehat D)_+\operatorname{softplus}(z),
$$

$$
g=\frac{\partial\ell}{\partial z}
=|\widehat D|\left(p-\mathbf1[\widehat D>0]\right).
$$

因此，对 $\widehat D<0,p\approx0$：

$$
g=|\widehat D|p\approx0.
$$

这表示当前动作已经符合监督。**若各状态有独立参数，这些正确状态不会因为另一状态的 BCE 更新而移动。** 共享 VLM 才引入了跨状态传播。[生产 loss：`decision_loss.py`](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/pipeline/grpo/v52_dl/decision_loss.py)

我还用公开的 64 个 audit 状态做了一个直接检查：

| 使用哪个模型的 $p$ | 正 $\widehat D$ 状态对平均 $g$ 的贡献 | 负 $\widehat D$ 状态的贡献 | 总平均 $g$ |
|---|---:|---:|---:|
| reference | $-0.00816407$ | 约 $+6.1\times10^{-12}$ | $-0.00816407$ |
| final main | $-0.00796332$ | $+0.02561438$ | **$+0.01765107$** |

梯度下降会减去 $g$。所以在最终模型上，**这些 audit 状态的总 BCE 梯度已经会压低统一的直答偏置**。不能由初始的 crop 侧小梯度，推断它在整个训练中一直没有纠正作用。

这仍只是 audit 上的代数检查。实际训练的 240 个状态、参数 Jacobian、RLOO 梯度及 Adam 状态没有被这张表恢复出来。

## 1.4 全局偏移确实存在，但一个常数不能解释全部变化

我拟合了一个只使用概率、不使用收益标签的描述模型：

$$
p_b(h)=\sigma\!\left(\operatorname{logit}p_{\mathrm{ctrl}}(h)+b\right).
$$

以 `p_main` 为 soft target 最小化交叉熵，得到：

$$
b=22.59035.
$$

但它对 `p_main` 的概率 RMSE 为 **0.13951**；完全不改变 control，RMSE 反而为 **0.10047**。而且 main 相对 control 出现 **222 对严格排序反转**，总计 2,016 个状态对。常数 logit 平移严格保序，所以：

> 数据支持“广泛共同上移”，不支持“全部变化只是统一加了一个常数”。

这个拟合不是历史训练的因果分解。换一种拟合准则，所谓“偏置部分”和“残差部分”的效用分摊还会变化，不能挑一个有利分解证明机制。

## 1.5 生产接线没有出现我们最容易猜测的那几类错误

读真实代码后，我没有发现“BCE 被错误埋进 token mean”或“先单独更新 BCE 再更新 RLOO”的明显问题：

- auxiliary loss 单独按预定状态数 $H_0$ 平均，没有除以 $\sum|\widehat D|$；
- $\widehat D$ 被 detach；
- auxiliary backward 累加在同一次 optimizer step 之前；
- 原 RLOO 仍按 response token 聚合；
- 标签 log-prob 使用多 token **求和**；
- 不可用状态仍留在 $H_0$ 分母中。

但是，**标签 mask 只限定直接计分位置，不隔离参数影响**。实际配方采用共享模型参数、联合裁剪及 fresh AdamW。当前单次 on-policy 更新路径的 PPO ratio 数值为 1，clipping 也没有提供我们希望的动作漂移约束。[adapter](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/pipeline/grpo/v52_dl/easyr1_caption.py)；[启动配方](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/pipeline/grpo/scripts/grpo_dl.sh)；[EasyR1 本地补丁](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/pipeline/grpo/easyr1_602a128_local.diff)

一个有用的局部表达是：

$$
\Delta z(h_j)\approx
-\eta J_j^\top P_t
\left[
g_{\mathrm{RLOO}}
+\frac{\beta}{H_0}\sum_i g_iJ_i
\right],
\qquad J_i=\nabla_\theta z(h_i).
$$

这里 $P_t$ 仅表示优化器预条件作用的近似。它说明：少数状态的更新可以通过共享参数改变大量其他状态；$\beta$ 小不等于实际函数变化必然小。

**上一版被挑战的关键假设正是：局部标签监督、较小系数和短训练，足以产生有用且相对局部的路由变化。** 上一版虽然已经提醒共享参数与 $K$ 噪声风险，但实验没有支持这个实施上的押注。

---

# 二、重新区分 advantage、credit、目标与噪声

## 2.1 这次不再把“保持原期望梯度”作为硬约束

| 操作 | 数学对象 | 合理承诺 |
|---|---|---|
| 状态 baseline | $(R-b(h))\nabla\log\pi$ | 满足采样、时间顺序和 reduction 条件时，改变估计方差 |
| 合法 advantage | $Q(h,a)-V(h)$ | 可以依赖动作；动作依赖本身不是错误 |
| fc 的 token 系数替换 | 对 crop 轮整段 token 改写优势 | 改变训练向量场，不能自动叫原目标的无偏估计 |
| reward shaping | 改奖励再构造 return | 一般改变目标；特定势能形式有边界条件 |
| 动作 BCE | 用收益差建立分类 surrogate | 显式改变监督对象与梯度，是否有用应实验检验 |
| 本次方案 | 估计动作价值差，再做受限策略改进 | 保持参考续跑固定，直接检验选路是否改善 |

合法的新目标或新 surrogate 并不天然错误。需要避免的是：**改变了优化对象，却继续使用旧目标的无偏性、因果信用或保策略结论来背书。**

fc 的真实代码把

$$
\hat v(s_{t+1})-\hat v(s_t)-\lambda
$$

写入 crop 轮整个 assistant span；zero 在相同位置写入 $-\lambda$。这与把势能项加进 reward、再正确计算 return-to-go 不同。[fc 实现](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/pipeline/grpo/v51_fc/fourcase_credit.py)；[zero 实现](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/pipeline/grpo/v51_fcz/fourcase_credit.py)

若采用势能塑形，则必须有：

$$
F_t=\gamma\Phi(s_{t+1})-\Phi(s_t),\qquad
\sum_t\gamma^tF_t=-\Phi(s_0)+\gamma^T\Phi(s_T).
$$

终止势能为零时才消去尾项。这个恒等式不等于“有限训练预算下会更少裁图”。**本次不选势能路线。**

## 2.2 有限 $K$ 的 BCE 变形是真问题，但还不是本次失败的完整归因

固定状态和当前 $p$，令：

$$
\delta=\mathbb E[\widehat D],\qquad
\eta_D=\mathbb E|\widehat D|.
$$

则：

$$
\boxed{
\mathbb E[g_{\mathrm{BCE}}]
=\eta_D\left(p-\frac12\right)-\frac{\delta}{2}
}
$$

以及：

$$
\mathbb E[\ell_{\mathrm{BCE}}]
=
|\delta|\operatorname{BCE}(\mathbf1[\delta>0],p)
+
\frac{\eta_D-|\delta|}{2}
[-\log p-\log(1-p)].
$$

第二项是**均匀目标的交叉熵项**，不是 Shannon entropy bonus。它会将概率拉向中间，强度取决于分叉噪声。

因此，“收益均值无偏”不推出“符号与绝对值变换后的 BCE 梯度无偏”。但它是否主导了实际训练，还需要训练状态上 $(p,\widehat D)$ 的联合记录及分叉结果。公开的 audit 不能替代这些数据。

两个明确反例，均取每分支 $K=4$、crop 一次：

| 真实成功率与当前概率 | 真收益差 $\delta=q_A-.99q_C$ | 期望 BCE 梯度 | 结果 |
|---|---:|---:|---|
| $q_A=.50,q_C=.51,p=.01$ | $-.0049$ | $-.13153346$ | 提高实际上较差的 answer 概率 |
| $q_A=q_C=.50,p=.90$ | $+.005$ | $+.106875$ | 降低实际上较好的 answer 概率 |

这些数值来自全部 25 种二项成功次数组合的枚举，说明机制在数学上可能发生，**不证明实际 240 个训练状态处于这个区域**。

对于 $.01$ 成本差也不能一概而论：

- 两边都确定答对时，$D=.01$，$K=4$ 没有成功率采样噪声；
- 两边成功率都约 .5 时，$D$ 的标准差约 **.352**，成本引起的均值差仅 **.005**。

所以“4 分叉太弱”既不是统一结论，也不能推出“增加分叉即可修复”。

## 2.3 为什么不选几个看似直接的 loss 修补

| 候选 | 梯度或最优点 | 本次不采用的原因 |
|---|---|---|
| $-\widehat D z$ | $g=-\widehat D$ | 对非零 $D$ 无下界；持续推动 logit，没有参考保护 |
| 绝对回归 $z\approx\widehat D/\tau$ | $g=\tau z-\widehat D$ | $D=0$ 时把已有置信度拉回 $z=0$，并非只修正错误 |
| 相对回归 $\frac\tau2(z-z_0)^2-\widehat D(z-z_0)$ | $g=\tau(z-z_0)-\widehat D$ | 有界且保留参考，但共享参数干扰仍在，小步又可能完全够不到稀有动作 |
| 直接收益 $-\widehat D\sigma(z)$ | $\mathbb E g=-\delta p(1-p)$ | 没有上述 $\eta_D$ 项，但当前极端置信度下梯度非常小 |

相对更新的单状态最优点：

$$
z^*=z_0+\delta/\tau
$$

与 KL 正则策略改进的二动作形式相对应，是标准方法，不需要另造概念。[proceedings.mlr.press](https://proceedings.mlr.press/v97/geist19a/geist19a.pdf?utm_source=chatgpt.com)

问题在于实际量级：三个尚未直答、但 audit 中 $\widehat D>0$ 的状态，参考直答概率约为

$$
4.58\times10^{-14},\quad
3.57\times10^{-13},\quad
6.80\times10^{-12}.
$$

要把它们推过 $p=.5$，需要约 **25.7–30.7** 的 log-odds 变化。一个 $\pm2$ 或 $\pm4$ 的“小参考步”几乎不会改变其行为。

因此，我选择改变**可执行的策略参数化**，而非继续寻找一种更大的原模型 logit 梯度。

---

# 三、复用哪些卡，以及它们真正提供了什么

主方案只采用两张主卡；其余已读卡用于排除不合适的方向。

| 主卡 | 对象／可见信息 → 比较 → 信号 → 梯度接收者 | 对本方案的预测与反证 | 借鉴边界 |
|---|---|---|---|
| **K19：MC 步标签** | 同一真实前缀 → 两种完整续跑 → 有符号收益差 → 小型价值差预测器 | 如果前缀特征包含可泛化的动作价值信息，独立题上应胜过一维阈值；不胜过则放弃这套估计与函数类 | MM-PRM 支持从前缀续跑构造监督；它不是本方案的两动作因果贡献证明 |
| **K27：限制更新范围** | 冻结表示与续跑能力 → 只改变路由 → 独立选路收益 → 新增小头 | 在同一 $h,a$ 下，后续计算应保持相同；若仍伤害总体准确率，伤害来自选路或接口，而非新 loss 修改了答题参数 | 原文的低学习率／LoRA 结果是经验依据；本方案采用更强的冻结条件，不能把它说成原文定理 |

MM-PRM 的相关方法确实从前缀继续采样，以完成任务的成功频率形成监督；这测量的是特定续跑策略的可达成功率。K27 对应原文讨论了更新范围、学习率及能力保留，但并不保证 LoRA 或小学习率没有干扰。[arxiv.org](https://arxiv.org/html/2505.13427v2?utm_source=chatgpt.com)

两个实际采用的 pattern 是：

- **`constrain_what_changes`**：把“选择哪条路线”与“路线内如何回答”分开；
- **`measure_example_utility`**：用独立收益检验新增状态信息是否值得，而不以训练 loss 或直答率验收。

我没有采用 K29 的小幅 logit 一致性约束，因为当前置信度下存在明显的可达性问题；其原文的 logit replay 来自持续学习分类任务，迁移到生成式 VLM 只能算类比。也没有采用 K57 的外部 inspector：它增加另一套回答权限和推理成本，当前并非必要。[proceedings.mlr.press](https://proceedings.mlr.press/v199/bhat22b/bhat22b.pdf?utm_source=chatgpt.com)

**最近的标准先例是 approximate policy iteration / classification-based policy iteration。** 对同一状态估计各动作的 $Q$，再学习更好的动作选择，本来就是这个框架；保守混合也有成熟先例。当前值得检验的是具体任务中的效用，而非算法名称的新颖性。[arxiv.org](https://arxiv.org/html/1407.0449v1?utm_source=chatgpt.com)

---

# 四、唯一主方案：冻结续跑，预测价值差，有限概率接管首决策

## 4.1 目标、reference 与适用范围

固定现有 SFT-A，以及现有 **r1@40** 参考模型 $\pi_0$。不从三个失败分支继续，不修改 SFT，不在这个 MVE 中继续全参数 RLOO。

沿用收益：

$$
R=Y(1-.01N).
$$

这里 $N$ 是整条轨迹的裁图次数，invalid、截断等按统一严格规则计分。这个回报本身允许准确率与正确轨迹成本交换：

$$
J_R=P(Y=1)-.01\,\mathbb E[YN].
$$

**它不等于“保证准确率不降”。** 本方案只产生一个候选策略；保准确率、少裁图和实际推理节约是独立验收条件。不能因为 $J_R$ 上升就自动宣布达成用户目标。

本 MVE 只改**首个实际 caption 后、动作标签前**的决策。它不能解决“caption 已经写偏”或所有后续裁图决策，范围需要明确。

## 4.2 精确定义状态与三个续跑过程

状态 $h$ 包括：

- 原始 token prefix；
- 图像、crop 图像及 grid／位置绑定；
- 环境状态、剩余预算；
- 当前冻结模型的前缀表示。

状态纳入规则只能依赖动作前缀，不能根据后续答对、格式是否合法或是否裁图来筛选。现代码需要看到下一个 `<` 才确认边界；新接口应在预先约定的、token 对齐的 caption 结束位置暂停，不能偷看动作再决定是否纳入。

定义：

$$
P_{\mathrm{nat}}^0(\tau\mid h)
$$

为原样自然续跑；

$$
P_A^0(\tau\mid h),\quad P_C^0(\tau\mid h)
$$

分别为强制 `<answer`、`<grounding` stem 后，使用同一个冻结模型续跑至终止。

于是：

$$
Q_A(h)=\mathbb E_{P_A^0}R,\quad
Q_C(h)=\mathbb E_{P_C^0}R,\quad
\delta(h)=Q_A-Q_C.
$$

**测量对象是两条可执行路线的收益差。** crop 路线同时包含新视觉信息、额外推理和后续 recovery；这里不把它称为纯视觉信息的因果贡献。

### 接口上的重要修正

当前真实打分对象是 `<answer` 与 `<grounding`，分别为 2、3 个 token；闭合 `>` 没有包含在标签内。应称为**完整 stem 概率**，不能称完整合法动作概率：

$$
s_a=\sum_j\log\pi_0(t_{a,j}\mid h,t_{a,<j}),
\qquad z_0=s_A-s_C.
$$

必须求和，不能 token mean，也不能只取共有的第一个 token。

$p_0=\sigma(z_0)$ 只是两种 canonical stem 的相对概率，不自动等于自然直答率。当前 scorer 与采样的 image-token bias 也尚未完全对齐；新实现须使用相同温度与 logits processor。v5.2i 首轮校验已放开两种动作，但 SYSTEM 文本仍沿用旧 WORD 对应关系，这也要如实保留在测量契约中。[预注册](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/pipeline/grpo/v52_dl/PREREG.md)；[协议](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/pipeline/grpo/v52_dl/caption_v52i.py)

## 4.3 学习有符号价值差，不把它直接变成带噪 BCE 标签

每个训练状态，两分支各采样 $K=4$：

$$
\widehat D_i=
\frac14\sum_{k=1}^4 R_{A,ik}
-\frac14\sum_{k=1}^4 R_{C,ik}.
$$

取冻结模型在边界上的 hidden vector，归一化、按训练集居中，构造固定特征 $x(h)$。只拟合一个线性头：

$$
f_\phi(h)=b+w^\top x(h).
$$

损失为：

$$
\boxed{
L_{\mathrm{gap}}
=
\frac1{2H_0}\sum_i
e_i\left[f_\phi(h_i)-\operatorname{sg}(\widehat D_i)\right]^2
+\frac{\alpha}{2}\|w\|^2
}
$$

其中：

- $e_i$ 为仅由前缀决定的可用标记；
- 不可用状态贡献零，但仍保留在预定 $H_0$ 分母；
- 分叉失败、非法或截断按奖励规则计分，不能删掉；
- $D=0$ 的可用状态保留；
- 截距不惩罚，特征及标签全部 stop-gradient；
- 原 VLM 的所有参数冻结，**所有 LM token loss mask 为零**。

这不再是“加到 RLOO 上的另一项 token loss”，而是明确改为一个受限的策略改进实验。

为避免再调一组正则超参，预锁：

$$
C=\frac1{H_0}\sum_i e_ix_ix_i^\top,\qquad
\alpha=\operatorname{tr}(C)/16.
$$

它只由无标签特征定尺度，满足斜率部分的有效自由度：

$$
\operatorname{tr}[C(C+\alpha I)^{-1}]\le16.
$$

截距另计；若 $\operatorname{tr}(C)=0$，停止条件头方案。可以在 CPU 上直接解 ridge，不需要神经网络训练循环。

### 梯度及 $K=4$ 的边界

$$
\partial_{f_i}L
=\frac{e_i}{H_0}(f_i-\widehat D_i),
\qquad
\boxed{\partial_{z_0}L=0}.
$$

后一个零是因为原模型被冻结，不是因为设计了某个特殊 token mask。

对固定 $f,h$：

$$
\mathbb E[(f-\widehat D)^2\mid h]
=(f-\delta)^2+\operatorname{Var}(\widehat D\mid h).
$$

因此噪声增加回归方差，没有 BCE 中额外的均匀目标交叉熵项。**这不消除有限样本过拟合，也不保证 hidden features 含有所需信息。**

候选动作取：

$$
q_\phi(h)=\mathbf1[f_\phi(h)>0],
$$

即 $f>0$ 候选 answer，否则候选 crop；精确平局固定选 crop。

它的单状态误选损失满足：

$$
|\delta|\,
\mathbf1[\text{候选动作非最优}]
\le |f-\delta|.
$$

这说明回归误差与决策损失有直接联系；不代表回归已解出带准确率约束的最优策略。

## 4.4 用真实随机混合限制改动，不再推动原模型 logit

部署时只在第一个可用 $h$ 抽一次开关，预锁 $\epsilon=.1$：

$$
\boxed{
P_\epsilon^q
=(1-\epsilon)P_{\mathrm{nat}}^0
+\epsilon[q(h)P_A^0+(1-q(h))P_C^0].
}
$$

含义是：

- 90% 原样自然续跑；
- 10% 按候选门控选择 stem，之后仍由冻结模型续跑；
- 后续轮次不再接管；
- 不可用状态全部自然回退。

**这个混合必须真实随机执行，不能再对混合概率取 argmax。** 否则在原模型几乎必裁图时，10% 的接管权可能完全不改变实际行为。

它不要求 $p_0P_A+(1-p_0)P_C=P_{\mathrm{nat}}$，因此避免把 stem 归一化概率误当自然策略。

由于干预前的状态分布相同，干预后全部固定，得到精确关系：

$$
\boxed{
J(P_\epsilon^q)-J(\pi_0)
=
\epsilon\,
\mathbb E_h\!\left[
e(h)\{qQ_A+(1-q)Q_C-V_{\mathrm{nat}}\}
\right].
}
$$

这是本次重设计最有用的性质：**共同状态的路线比较，可以对应这个实际部署策略的完整期望收益变化。**

同时必须保留边界：

- $\epsilon$ 限制的是绝对干预概率，不是直答比例；
- 最坏准确率变化仍可达 $\epsilon$，所以它不证明保准确率；
- 冻结保证同 $h,a$ 下续跑参数不变，但提前回答仍可能让必要 recovery 根本没有机会发生；
- 新小头仍跨状态共享参数，可能学成全局偏置；这要由下面的强对照和独立评测否定，不能凭结构自证。

---

# 五、唯一匹配对照：充分校准原动作分数的一个阈值

控制使用完全相同的：

- 参考模型、真实状态；
- $K=4+4$ 分叉缓存；
- $\epsilon=.1$ 混合接口；
- 预算、严格评分及测试题。

它不读取新的 hidden-state 线性组合，只对原分数 $z_0(h)$ 加一个全局偏置后选动作：

$$
q_b(h)=\mathbf1[z_0(h)+b>0].
$$

**不要用稀疏的 $\{0,\pm16,\pm32\}$ 网格把对照做弱。** 将训练集的 $z_0$ 排序，枚举所有相邻分数之间的阈值，以及“全部 answer／全部 crop”两个端点，选择：

$$
b^*=\arg\max_b
\frac1{H_0}\sum_i e_iq_b(h_i)\widehat D_i.
$$

平局按预锁规则选择较少 answer 的阈值。测试标签不参与选择。

主、控都是硬候选动作，通过同一随机混合执行，因此：

$$
\boxed{
J_{\mathrm{main}}-J_{\mathrm{ctrl}}
=
\epsilon\,\mathbb E_h[
e(h)(q_\phi-q_{b^*})\delta(h)
].
}
$$

这是一个很强的一维校准对照：它可以利用原模型已有的状态排序，但不能重新排序。若小头不能胜过它，就没有证据支持新增状态模型的价值。

这个比较检验的是“新增状态表示与价值估计流程”，不是一个纯粹的 loss 形式消融。共同自然模型 $\pi_0$ 是必要参照，不增加第三个训练臂。

---

# 六、四种情况的手算与两个必须保留的反例

以下假设 crop 恰好一次，展示**观测标签**，并取预测头当前 $f=0$。为简洁省略 $1/H_0$。

| answer / crop 成功数 | $\widehat D$ | 回归梯度 $f-\widehat D$ | 候选更新方向 |
|---|---:|---:|---|
| $0/4,\;4/4$ | $-.99$ | $+.99$ | 支持 crop |
| $4/4,\;4/4$ | $+.01$ | $-.01$ | 小幅支持 answer |
| $0/4,\;0/4$ | $0$ | $0$ | 没有方向性证据；平局候选 crop |
| $4/4,\;0/4$ | $+1$ | $-1$ | 支持 answer |

极端置信度下：

- 若原模型 $p_0\approx0$、真实应 crop，新方案不要求继续把其 logit 压得更低；
- 若原模型 $p_0\approx10^{-14}$，但可靠估计支持 answer，新小头可以在接管的 10% 轨迹上选择 answer，无需将原 logit 推高 30；
- 若已经 $f=\delta$，预测头梯度也应趋于零。**“正确后梯度小”仍然是正常行为。**

### 反例一：两边都失败，不代表 crop 没有价值

若：

$$
q_A=.01,\qquad q_C=.10,
$$

则：

$$
\delta=.01-.99(.10)=-.089.
$$

但两边 $K=4$ 全失败的概率：

$$
(.99)^4(.9)^4\approx .63025.
$$

约 63% 的批次会得到零标签。回归也不能从不存在的成功样本中恢复真值。且若当前 $f\ne0$，零标签会把它向零收缩，不能描述成“完全不影响必要探索”。

保护来自有限接管、冻结续跑和真实评测；不是宣称已识别所有必要失败探索。

### 反例二：一次同对，不代表冗余

若：

$$
q_A=.90,\qquad q_C=.99,
$$

即使某一对样本都答对，仍有：

$$
\delta=.90-.99(.99)=-.0801.
$$

crop 在期望上明显更好。反过来，观察到一次“错→对”也可能只是随机波动。

### 延迟 recovery

若现在答必错，但先 crop、再恢复性 crop 可以成功：

$$
Q_A=0,\qquad Q_C=.98,
$$

整段续跑价值差会支持首次探索。这是它相对即时 answer-now 差值的优势。若冻结续跑本身没有掌握恢复能力，该价值也不会凭估计产生。

---

# 七、最低成本、可证伪的 Probe

## 7.1 现在先做零 GPU 工作；公开材料还不足以跳过这一步

需要取回已经存在的文件：

- `audit_states.json`、三个 `score_*.json`；
- 每状态每个 $k$ 的原始 fork 结果；
- 原始评测轨迹、阶段时间与 token 账；
- 服务器实际导入的共享模块路径及 hash。

公开树缺少这些原文件。生产训练 fork 日志也没有保存全部 prefix IDs，所以即使补到 rewards，也未必能恢复真实训练梯度。

### A. 固定的 2+2 分叉检查

预先固定 $k=\{0,1\}$ 和 $k=\{2,3\}$：

$$
D_L=\bar R_{A,L}-\bar R_{C,L},\qquad
D_R=\bar R_{A,R}-\bar R_{C,R}.
$$

在条件独立采样下：

$$
\mathbb E[D_LD_R\mid h]=\delta_h^2,
$$

$$
\mathbb E\!\left[\frac{(D_L-D_R)^2}{4}\mid h\right]
=\operatorname{Var}(\widehat D_{K=4}\mid h).
$$

而且逐样本有：

$$
\widehat D_4^2
=
D_LD_R+\frac{(D_L-D_R)^2}{4}.
$$

报告跨题均值及区间，**不要把负的 $D_LD_R$ 截成零**。

对独立 audit 再计算：

$$
U_L=\overline{(p_{\mathrm{main}}-p_{\mathrm{ctrl}})D_L},
\qquad
U_R=\overline{(p_{\mathrm{main}}-p_{\mathrm{ctrl}})D_R}.
$$

这能检查负效用是否在两半分叉中重复出现。训练已经使用全部 K4，不能事后把其中 K2 称为训练的独立验证集。

### B. 改善空间检查：目前已有一个不乐观结果

我在公开 64 态上枚举了最佳单阈值，并与“将每个 $\widehat D$ 当真值、允许逐状态完美选择”比较：

$$
\max_{q(h)\in\{0,1\}}\overline{q\widehat D}
-
\max_b\overline{q_b\widehat D}
=
\boxed{.0081640625}.
$$

应用 $\epsilon=.1$ 后仅为：

$$
\boxed{.00081640625}.
$$

这只是**开发表格上的样本内空间**，不是总体真值上界；最大化带噪标签还会产生选择偏差。但它说明：首决策上的可见机会很少，不能仅凭换了模型形式就值得再花 GPU。

同时必须用原始 $Y,N$ 检查这些机会是否只是用准确率换成本，以及节省是否足以覆盖门控和 stem scoring 开销。

### C. 精度检查必须先于 GPU

新方案相对自然模型的估计需要 $V_{\mathrm{nat}}$。每题原始完整 rollout 的 suffix 可以提供一个无偏样本，前提是状态筛选不依赖未来结果。

但这会引入自然回答噪声。例如，两路与自然回答都近似成功率 .5、每路 K4、128 题时：

$$
\operatorname{SE}(\widehat{\Delta J})
\approx
.1\sqrt{\frac{.25/4+.25}{128}}
\approx .00494.
$$

这是示例量级，不是实际方差断言；它已经远大于上面的 .000816。减小 $\epsilon$ 会同时缩小信号和标准误，不会自动改善可检出性。

**如果补齐原始数据后，可见空间仍如此小、预算内区间无法回答问题，就停止这个首决策方案，不增加 K、不换阈值补过关。**

## 7.2 仅在上述检查支持继续时：一个小 MVE

不做第四轮全参数 RLOO。只做一次固定参考模型的分叉采集、一次 CPU 小头拟合、一次阈值选择。

| 项目 | 上限 |
|---|---:|
| 128 个训练题、128 个封存测试题，各一条自然参考轨迹 | 256 |
| 每题 answer/crop 各 K4 | 2,048 |
| 64 个封存测试状态的 E2：额外 crop 分支 K4 | 256 |
| 同 64 状态的一次自然 E2 续跑 | 64 |
| 实际接口核查：16 个首决策态＋16 个共同错框后态，两臂各一次 | 64 |
| **完整轨迹／终止续跑合计上限** | **2,688** |

E2 在首次 crop 才施加扰动，所以 answer 分支可以复用；共同错框后核查只验证冻结续跑及关闭后续门控的实现，不将几十条记录当作恢复非劣证明。

能证明来自同一参考模型、保存了完整状态的现有缓存可以抵扣采集。旧 240 个在线训练状态来自变化中的策略，不能直接混成固定 $\pi_0$ 的标签库。

**全流程硬帽：2 GPU 小时、50 万 generated tokens、1,200 万 processed tokens，任一先到即停。** 包含模型加载、图像预填、hidden 提取、标签 scoring、失败重试和全部核查。公开旧 fork 平均约 85.6 generated tokens，只可作预算参考，不可保证新流程成本。[现有计数](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/ba449babb4072a067f63b46419a169903d0cf407/logs/dl/compare_dl.json)

第一批技术计时也计入预算；若预测无法完成锁定集合，报告预算不满足。不能靠删掉较长题目或临时缩 K 完成“通过”。

测试集合必须排除：

- 已反复用于设计的 E1；
- 已查看及疑似重叠的全部 40 题；
- SFT/RL 训练泄漏及图像近重复。

按路径或文件名排重不够。找不到真正隔离的集合，就只能做开发诊断。

## 7.3 精确证据与停止规则

封存测试题上，主头与阈值均已固定。用同一批分叉计算：

$$
\widehat G_{\mathrm{main-ctrl}}
=
\frac{\epsilon}{H_{\mathrm{test}}}
\sum_i e_i(q_{\phi,i}-q_{b,i})\widehat D_i,
$$

以及：

$$
\widehat G_{\mathrm{main-nat}}
=
\frac{\epsilon}{H_{\mathrm{test}}}
\sum_i e_i[
q_{\phi,i}\widehat Q_{A,i}
+(1-q_{\phi,i})\widehat Q_{C,i}
-R_{\mathrm{nat},i}
].
$$

把 $R$ 替换为 $Y,N$、生成 token 和处理成本，可得到对应差值。按**题**成对 bootstrap；同题的自然 suffix、所有 fork 及两策略权重一起重采样，不能把 K4 当四倍题数。

| 观测 | 决定 |
|---|---|
| 主方案不能在独立题上胜过最佳单阈值 | 不支持新增状态头；停止该方案 |
| 主控效用区间上界低于 0 | 直接反对该方案 |
| 只胜过阈值，却没有相对自然模型的净改善 | 不推进；避免仅因控制差而显得有效 |
| 裁图少了，但准确率有明确伤害 | 失败，不以成本收益抵消 |
| 净推理开销没有减少 | 未达到成本目标 |
| 区间无法区分 | 未决，到预算停止；不解释为不降准确率 |
| 边界、图像绑定、合法性、冻结参数或 reduction 不符 | 实现未对应算法，不进入科学验收 |

若要宣称达成“保准确率少裁图”，需要 clean/E2 准确率差的区间支持预先声明的非劣要求。**本方案不把负 2 点偷偷当作用户已经接受的损失。** 小样本只能排除某些较大伤害时，就如实写“准确率仍未确认”。

E1 不再被用于调正则、挑阈值或换 $\epsilon$。旧 DL 的失败结论也不因新 Probe 改写。

## 7.4 可直接交给执行者的核心伪代码

```python
# 提案；本轮未执行
pi0 = freeze(existing_r1_step40)
epsilon = 0.1
K = 4

# train/test 问题、slot、RNG namespace 先锁定
for question in locked_questions:
    natural = rollout_to_terminal(pi0, question)
    h = extract_first_prefix_only_boundary(natural)

    if not eligible_from_prefix(h):
        save_ineligible_with_zero_difference(question)
        continue

    # 同 h、同图像绑定、同剩余预算、严格评分
    QA = mean(strict_reward(fork(pi0, h, "answer", k)) for k in range(K))
    QC = mean(strict_reward(fork(pi0, h, "crop",   k)) for k in range(K))

    save(h, detached_hidden(h), QA - QC, natural.reward, all_fork_records)

# 只用训练部分
head = fit_one_centered_ridge_gap_model(alpha=trace(C_train) / 16)
threshold = best_training_threshold_by_sorting(reference_stem_scores)

# 实际部署；每条轨迹仅接管首个可用决策
def continue_policy(h, arm):
    if not eligible_from_prefix(h) or bernoulli(epsilon) == 0:
        return natural_continue(pi0, h)

    if arm == "main":
        action = "answer" if head(h) > 0 else "crop"
    else:
        action = "answer" if reference_score(h) > threshold else "crop"

    return force_stem_and_continue_frozen(pi0, h, action)
```

现 adapter 一次生成整轮，需要新增准确的暂停／续跑接口。不能把该修改描述成“只换 loss、部署零开销”。

---

# 八、给导师的简短解释

> 这次不再要求已经正确、很自信的裁图状态持续产生大梯度。三轮结果提示，先要把“选择哪条路线”与“整台模型怎样回答”分开检验。我们固定现有 RLOO 基座及其后续能力，用同一真实状态的两路续跑估计收益差，只训练一个小预测头；部署时仅以预定小概率让它接管首个动作。唯一对照充分校准原动作分数的一个全局阈值，双方共用全部分叉。若新状态信息不能在独立题上胜过该阈值，或省裁图伴随准确率伤害，就停止；若现有数据表明首轮改善空间太小，也不启动新 GPU 实验。这是一次标准策略改进的受限验证，不宣称发现了新的工具信用理论。

---

# 九、实际已读与未读材料

## 已直接读取、用于本答的材料

- **新任务入口**：`5c63a280…/decision_loss_redesign_20261022/README.md`。
- **报告**：`6c2bddf…/docs/30_rl_credit_report_20261022.md`，§7–§7.5。
- **真实 DL 实现**：`ba449babb…` 下的 `PREREG.md`、`decision_loss.py`、`easyr1_caption.py`、`caption_v52i.py`、`dl_compare.py`，以及 `dl_audit.py` 的状态生成、分叉、scoring 和统计相关代码。
- **真实结果**：`logs/dl/compare_dl.json` 全部 64 个状态及汇总；fc/fcz 比较 JSON。
- **训练接线**：`grpo_dl.sh`、EasyR1 本地补丁；fc/fcz 的 `fourcase_credit.py`、fcz 预注册；DL 单测源码。**读过单测源码不等于运行过测试。**
- **上一版材料**：`PRO_RESPONSE.md` 全文、`MENTOR_INTERPRETATION.md`、交接 README。
- **完整卡片六张**：K11、K19、K27、K29、K36、K57。主方案采用 K19、K27。
- **完整 parent pattern 两篇**：`constrain_what_changes`、`measure_example_utility`；另读了 parent/subpattern 总览。
- **关键原文相关方法段**：MM-PRM `2505.13427v2`；*Fine-tuning MLLMs Without Forgetting Is Easier Than You Think* `2603.14493v1`；CAPI `1407.0449v1`；TRPO 2015；Geist 等 2019 正则化 MDP；K29 对应 PMLR 原文的算法与 logit consistency 段；VideoSEAL `2605.12571v1` 的 planner/inspector 方法段。这里没有把“读到关键方法”扩写成“全文所有实验均已审核”。

并行代码审查还补核了指定 EasyR1 `602a12820f69cf45ea98de08a63b3509313a7d02` 的 actor、FSDP、config 与 rollout，以及 fc/fcz 的其他 adapter 路径；我没有在本轮逐行重读全部上游文件。

## 未取得或未完整重读的材料

- 未取得原始 `audit_states.json`、`score_*.json`、逐 fork JSONL、完整评测轨迹、训练 optimizer 状态、checkpoint/tokenizer 实体及全流程时间账。故没有执行实际 2+2 拆分，也没有重放历史训练梯度。
- 公开目录存在共享 reward/contracts 等依赖缺口；没有服务器的 `module.__file__` 与 hash，不能确认其实际导入版本。
- 其余 **52 张机制卡、12 篇 parent pattern、19 篇 subpattern 全文**未逐篇读取；读索引不等于读完整库。
- 本轮没有重新完整阅读 K11 的 VisionThink/AdaTooler、K36 的 RLHF-V，以及 CounterCredit/TACO 原文。未据此声称本方案相对它们具有新颖性。
- Kakade–Langford 2002 原始 CPI PDF 的正文解析不可靠，未将其标为已读通；保守混合的原文依据采用已读的 TRPO 相关部分，并直接推导本方案的一次干预恒等式。
- 未重新通读历史 docs/27–33 与所有旧 dV/DB 实验；本次依据新入口要求核查三轮真实实验及上一版设计，没有用旧对话记忆补写缺失结果。
