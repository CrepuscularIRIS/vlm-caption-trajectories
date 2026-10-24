# 网页 Pro 完整设计回复

收集日期：2026-10-22。模式：Pro。页面已显示 **Response complete**。输入材料固定于 `65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc`；[实际提示词](SENT_PROMPT.md)。

以下为完整回复，仅将数学定界符转为 GitHub 支持的 `$` / `$$`；未删改推导和结论。**这是 Pro 的设计与审查意见，不代表 Codex 已独立验收全部论断或获准启动实验。** 其已读/未读范围见第 9.4 节。原回复 UTF-8 SHA256：`55df6e12318ce50ad1816f971b5ffdad02a1a1e1f4a4fc5b488d48fa526555cb`。

---

导师这次把问题收窄到了一个更明确、也更容易检验的对象：**模型已经看过当前图、写完这段 caption，现在应该直接回答，还是先裁图再继续？**

把这个时刻保存为同一个 $h$，让两条路线各自多跑几次，用最终收益比较它们，再直接训练动作标签的相对概率。这样，研究首先要解决的是“此刻怎样选路”，不必先给整段 caption、每个坐标和每次失败分别判定客观功劳。

**我的建议是保留导师的加权 BCE 原式，做一个受限的首次动作实验。它应被明确定位为“基于分叉估值的动作分类辅助目标”，而不是新的 baseline，也不是原 RLOO 梯度的无偏改写。** 但公开代码揭示了一个必须先解决的协议问题；数学推导又揭示了一个不能忽略的 K=4 效应：

1. **当前协议在 caption 之前就通过 `Step WORD` 锁定了合法动作。** 原协议下，同一个 caption 后的 $h$ 通常不能合法地走两条路线。
2. **K=4 的收益估计即使无偏，加权 BCE 的梯度也不是精确 Q 加权 BCE 的无偏估计。** 它会额外产生将两种动作概率推向接近各半的作用。
3. 因而，**“裁图减少”不足以证明学会了更好的工具取舍**。最小实验必须同时检查独立收益差上的选择改善、真实答题与成本，以及错框后的继续作答能力。

本轮只进行了资料阅读、源码审查和离散数学枚举，**没有修改训练代码、执行训练或访问训练 GPU**。

---

# 一、实际读取版本与独立诊断

## 1.1 本轮依据是指定的新提交

我读取的是：

**`65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc`**

提交说明为 `docs: publish action decision loss brief and VLM pattern library`。不是用旧对话或 `main` 的内容代替这个版本。

首先完整读取了：

- [README.md](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/docs/research_design/decision_loss_20261022/README.md)
- [PRO_DESIGN_BRIEF.md](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/docs/research_design/decision_loss_20261022/PRO_DESIGN_BRIEF.md)
- [MENTOR_INTERPRETATION.md](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/docs/research_design/decision_loss_20261022/MENTOR_INTERPRETATION.md)

其中 `PRO_DESIGN_BRIEF` 的 blob 为 `5a036cbbdf5e04ce3291861bedfd71150ad3aafa`，`MENTOR_INTERPRETATION` 的 blob 为 `7255c9788c255e5e3edc1576ee9d9c8b37f8ac64`。文末列出更完整的已读和未读范围。

## 1.2 最重要的生产代码事实：动作已经被 WORD 提前承诺

当前格式的顺序是：

```text
<think>
Step N | WORD
caption ...
</think>
<answer> 或 <grounding>
```

但 `WORD` 并非普通描述。生产校验要求：

- `locate / re_examine / fallback` 对应 `grounding`；
- `commit / abstain` 对应 `answer`；
- `abstain` 还要求回答 `UNCLEAR`。

v5.1 实际调用这些检查，正式 RLOO rollout 遇到检查错误就终止为 `invalid`；非合法回答终止不会获得正确奖励。证据链在 [caption_v51.py](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/code/reference/caption_v51.py)、[caption_v3.py](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/code/reference/caption_v3.py) 和 [正式 rollout 实现](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/pipeline/grpo/v51_cropfloor/easyr1_caption.py)。

因此，这三项目前不能同时成立：

> 同一个真实 caption 后的 $h$；不改现有协议；answer 与 crop 都是合法可执行动作。

例如，$h$ 已含 `Step 1 | locate`，强制 `<answer>` 后即使答对，也可能因协议冲突而得到零。此时测到的差异首先是**格式合法性的差异**，不能解释成直接作答的任务价值较低。

### 我选择的最小修订

**保留导师要求的 caption 后分叉位置，只修改首轮 WORD 的语义：**

> Step 1 的 WORD 记录初步意向；完成 caption 后允许修订选择，真实动作由最终 `<answer>` 或 `<grounding>` 决定。后续轮次暂时保持原协议。

相应地，首轮校验必须解除由 WORD 派生的动作绑定及相关 `abstain` 等硬约束；格式、步号、来源、坐标、预算和最终答案规则仍由统一执行器校验。

这是一项**明确的协议修改**，不能只在 fork 中偷偷放宽。正常 rollout、两个 fork、训练奖励和评测都必须采用同一版本。代码还断言数据中的 SYSTEM 与运行协议一致，所以需要同步版本化 SYSTEM 和初始输入前缀；**不需要重新训练或扩充 SFT-A**。

如果坚持完全不改协议，分叉就必须前移到 WORD 之前，重新生成相应 caption。那将是另一个估计对象，不再是本轮指定的“同 caption 后动作选择”。

## 1.3 当前证据支持什么

合并报告中的 r1，即 RLOO $\lambda=.01$，step160：

- E1：准确率约 $0.6635$，平均裁图 $2.078$；
- E2：准确率约 $0.6531$，平均裁图 $3.212$。

相对 RLOO0，裁图反而更多；准确率差的区间仍允许正负变化。这支持“当前配方没有实现减少裁图”，**不证明根因一定是信用路由，也不证明准确率已经保持不变**。[合并报告](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/docs/30_rl_credit_report_20261022.md)

我也读了 u39、u17 的文字记录并实际查看对应图片。图片中的目标数值可辨，但公开材料不足以让我独立重验完整题目和选项映射。这些病例说明“看到数字、组织推理、映射答案”可能分离，不能据此直接证明裁图冗余，更不能证明动作 BCE 能修复答案利用错误。

---

# 二、主算法：RLOO 加首次动作收益差 BCE

## 2.1 优化目标与边界

保留导师指定的任务收益：

$$
R(\tau)=Y(\tau)\bigl(1-\lambda N(\tau)\bigr),
\qquad \lambda=0.01.
$$

其中 $Y$ 是按统一严格规则判定的最终正确性，$N$ 是实际执行的 crop 数。

对应任务目标为：

$$
J_\lambda(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[Y(\tau)(1-\lambda N(\tau))\right].
$$

这里有一个必须写明的限制：

$$
J_\lambda
=
P(Y=1)-\lambda\,\mathbb E[YN].
$$

**它给成功轨迹的裁图定价，不直接惩罚失败轨迹的裁图，也没有直接给生成 token、视觉预填或 GPU 时间定价。** 我不把它改成此前的加性成本目标；实际成本是否下降，由评测决定。

新辅助损失使用同一种收益评估动作，但新增了一个分类代理目标。它与任务目标有关联，不能称为同一个目标的精确梯度。

## 2.2 算法契约

| 项目 | 主方案的具体定义 |
|---|---|
| 学生 | 固定 SFT-A 来源，不增加 SFT、不换基础模型、不加 critic/PRM |
| 监督位置 | **仅首次**实际 caption 后、动作开标签前 |
| $h$ | 原始 token 前缀、问题与选项、实际图像及绑定、当前 caption、环境状态、剩余预算 |
| 状态分布 | 每个训练问题预先随机指定一个主 rollout slot；从该 slot 取首次 $h$，不能按未来动作或正确性筛选 |
| 数量 | 保持每步 12 题、每题主 rollout $G=16$；最多取 12 个辅助状态 |
| answer 分支 | 强制完整 `<answer>` 开标签，之后由当前学生完成答案 |
| crop 分支 | 强制完整 `<grounding>` 开标签；bbox/source 自主生成，执行真实 crop，随后自主续跑到终答或失败 |
| 续跑策略 | 更新前当前策略 $\pi_k$，两分支相同采样配置，独立 RNG 流 |
| 分叉数 | 每个动作 $K=4$，不因平局或全错追加采样 |
| 原有边界 | 最多 8 次 crop、9 个 assistant turn；每 turn 640 token；总 response cap 沿用 7168 的真实 ledger 口径 |
| 真值 | 仅传给奖励函数，不进入模型可见 $h$、图像说明或 fork 提示 |
| 新监督接收位置 | 两种**完整动作开标签的全部 token**；不把 bbox、答案字母和分叉正文做成 SFT 目标 |
| 主 RLOO | 原参与训练的 assistant tokens 仍接收 RLOO 优势；环境观测和 scaffold 仍按原 mask 排除 |
| 更新方式 | 新鲜分叉、一次消费、与 RLOO 合并后做**同一次 optimizer step** |

预算必须从 $h$ 延续：首轮已经生成的 header、caption 和强制标签都占预算，不能在分叉后重新赠送 640 token。现有 cap 和 token/image ledger 的定义可见 [RLOO 启动脚本](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/pipeline/grpo/scripts/grpo_rloo.sh) 与 [caption_contracts.py](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/pipeline/grpo/v51_cropfloor/caption_contracts.py)。

**合法性失败、模型输出截断、任务预算耗尽计零；基础设施错误不应伪装成任务失败。** 后者应中止或以同一输入和种子重试，不能通过重抽直到成功来抬高某条分支的 Q。

预选 slot 如果没有形成可用的前缀，其辅助项记零、仍留在预定状态数的分母中，并报告覆盖率；不能根据后续输出是否有效重新挑一个有利 slot。

## 2.3 测到的究竟是什么

定义：

$$
Q_a^{\pi_k}(h)
=
\mathbb E\!\left[
R(\tau)
\mid h,\ \text{强制动作 }a,\ \text{随后执行 }\pi_k
\right].
$$

它测量的是：

> 当前学生在当前协议与预算下，从这个真实 caption 出发，选择某条路线的期望净回报。

其中 $Q_{\rm crop}$ 包含坐标选择、图像返回、更多推理、后续裁图、停止和 recovery。它**不是当前裁图的纯视觉信息贡献**。

另外，即使解除 WORD 的硬约束，caption 仍可能在语言上已经倾向裁图。固定这种 caption 后得到的 Q，只回答这个条件状态下怎样选路，不能代表模型在写出另一种 caption 后是否本可以直接回答。

---

# 三、完整损失、梯度、token mask 与 reduction

## 3.1 完整标签概率

令 $t_A,t_C$ 分别为两个完整开标签的 token 序列。前导空白属于 $h$ 还是标签，必须由真实 tokenizer 和原 token ledger 固定下来。

$$
s_a(h)
=
\sum_{j=1}^{|t_a|}
\log\pi_\theta(t_{a,j}\mid h,t_{a,<j}),
$$

$$
z_\theta(h)=s_A(h)-s_C(h),
\qquad
p_\theta(h)=\sigma(z_\theta(h)).
$$

这是**序列 log-prob 求和**，不是标签长度平均，也不是只取第一个 token。两个标签如果共享起始 token，只取那个 token 可能完全没有动作区分信号。

$p_\theta$ 是条件于这两个规范字符串的相对概率。另须记录：

$$
m_\theta(h)=e^{s_A(h)}+e^{s_C(h)}.
$$

$p$ 改善时，两个规范标签的总概率质量可能下降。与此同时，其他空白形式也可能合法，因此 $1-m$ 不能直接当作非法率；实际非法率要独立统计。

**采样配置也属于概率定义。** 当前 actor 使用 $T=.7$，而 rollout 还对 image token 施加了 $-100$ 的 logit bias。新增完整标签 scorer 应复现分叉采样器的固定 logits 变换及其顺序，不能默认 $T=1$，也不能只因为权重相同就声称分布完全相同。原 RLOO 分支维持既有计算路径，对已有差异如实记录。[actor 源码](https://github.com/hiyouga/EasyR1/blob/602a12820f69cf45ea98de08a63b3509313a7d02/verl/workers/actor/dp_actor.py)；[rollout 源码](https://github.com/hiyouga/EasyR1/blob/602a12820f69cf45ea98de08a63b3509313a7d02/verl/workers/rollout/vllm_rollout_spmd.py)

## 3.2 分叉标签与稳定实现

$$
\widehat Q_a(h)
=
\frac14\sum_{k=1}^4
Y_k^a(1-\lambda N_k^a),
$$

$$
D_h
=
\operatorname{sg}
\left(\widehat Q_A(h)-\widehat Q_C(h)\right).
$$

必须逐条先算 $Y(1-\lambda N)$ 再取均值。一般不能替换成：

$$
\bar Y(1-\lambda\bar N).
$$

因为成功与裁图数可能相关。

稳定的单状态损失为：

$$
\boxed{
\ell_h
=
(D_h)_+\,\operatorname{softplus}(-z_h)
+
(-D_h)_+\,\operatorname{softplus}(z_h)
}
$$

等价于导师的 $|D|\,\mathrm{BCE}(\mathbf1[D>0],p)$。$D=0$ 时严格没有辅助信号，不默认标成 answer。

其梯度是：

$$
\boxed{
\frac{\partial\ell_h}{\partial z_h}
=
|D_h|\left(p_h-\mathbf1[D_h>0]\right)
}
$$

以及：

$$
\nabla_\theta\ell_h
=
|D_h|(p_h-d_h)
\left(\nabla_\theta s_A-\nabla_\theta s_C\right).
$$

分叉回报和 $D$ 停止梯度。**辅助 loss 的计分位置仅限标签，不等于冻结前缀表示或其他行为。** 前缀仍参与计算标签概率，共享参数仍会影响 caption、bbox、答案和 recovery。

## 3.3 与实际 RLOO 相加

原 RLOO 为：

$$
A_i
=
r_i-\frac{1}{G-1}\sum_{j\neq i}r_j.
$$

当前生产实现把它复制到该轨迹的 response-mask=1 位置，使用全局主 rollout token 平均。记实际 PPO token loss 为 $\ell^{\rm prod}_{it}$，则新增目标应接为：

$$
\boxed{
L_{\rm total}
=
\frac{\sum_{i,t}m_{it}\ell^{\rm prod}_{it}}{\sum_{i,t}m_{it}}
+
\beta\frac1{H_0}\sum_{q=1}^{H_0}e_q\ell_{h_q}
}
$$

其中 $H_0=12$ 是预定状态数，$e_q$ 表示是否获得合法的前缀状态；平局、全零和不可用状态都不能通过缩小分母来放大其他样本。

**必须避免三种隐式改法：**

- 除以 $\sum_h |D_h|$：这会把本来很弱的 $0.01$ 成本信号重新放大。
- 把辅助 loss 塞进主 token loss 后，再乘主 microbatch token 数：这会使 $\beta$ 随轨迹长度和切片方式变化。
- 先做 BCE optimizer step，再做原 RLOO step：这破坏当前 on-policy shortcut 的更新前提。

我建议这个局部 MVE **固定 $\beta=1/256$**。这是一个公开的工程尺度选择：用约数百个生成 token 作为固定单位，将每状态一次的辅助损失与既有 token mean 接起来。它不是梯度范数相等的定理，也不是已经调好的最优值；不随 batch、$|D|$ 或结果动态调整。

$\lambda$ 决定任务收益中的裁图价格；$\beta$ 决定分类辅助梯度的强度。二者不能互相替代。

---

# 四、相同分叉信息的四种用途，以及 fc/zero 的位置

## 4.1 目标、估计器和梯度接收位置要分开

| 用法 | 最简形式 | 改变什么 | 能成立的主张 |
|---|---|---|---|
| 动作无关 baseline | $(R-b(h))\nabla\log\pi(a\mid h)$ | 梯度估计器 | 满足条件时不改变期望梯度，主要影响方差 |
| Reward shaping | $R'=R+\alpha F$ | 通常改变任务回报；特殊势能形式可保策略排序 | 必须说明新增偏好，或证明边界项满足条件 |
| Token advantage / weight | $\sum_t w_t c_t\nabla\log\pi_t$ | 梯度接收位置及权重 | 一般是新的训练 surrogate，不能自动继承原 PG 的解释 |
| 动作 BCE | $|D|\,\mathrm{BCE}(d,p)$ | 动作分类代理目标及直接监督位置 | 对分叉认为更优的动作施加明确监督，梯度不同于原 RLOO |

一个容易混淆的点必须纠正：

**动作依赖的优势本来就可以是正确的优势。**

$$
A^\pi(h,a)=Q^\pi(h,a)-V^\pi(h)
$$

依赖动作没有问题。有问题的是把一个依赖已选动作的扣除项，称为可以随意减去的“零均值 baseline”。

反过来，$b(h)=Q(h,\text{answer})$ 虽然评估了一个指定动作，只要无论实际选什么都减去同一个数，它仍可以是动作无关 baseline。

对于合法 baseline：

$$
\mathbb E_{a\sim\pi}
[b(h)\nabla\log\pi(a\mid h)\mid h]
=
b(h)\nabla\sum_a\pi(a\mid h)=0.
$$

所以它没有凭空增加新的停止偏好。它可能让有限训练更稳定、更快达到已有目标偏好的策略，但这与“识别并重新分配了工具的客观功劳”不是同一主张。

## 4.2 生产条件下，不能把这条等式直接照搬

我读了项目指定的 EasyR1 `602a128…`、项目补丁、DB 实现、`offline_forks.py` 和对应测试。审查结果如下。

| 条件 | 生产含义与判断 |
|---|---|
| turn 级 baseline | 必须在被加权 token 产生之前确定。同 turn 开始的状态可以；caption 后的 $b(h)$ 不能向前用于这段已生成 caption 并继续称动作无关 |
| 可变长度 token mean | 当前分母是采样得到的全局 response token 数，与动作及终止长度有关。零均值 baseline 在固定尺度下的证明，不能自动穿过这个随机分母 |
| PPO clipping | 当前补丁要求单 epoch、整批一次更新，并令 old log-prob 为 current detach；此时 ratio 数值为 1，clip 不提供实质限制。若改成复用数据或多次更新，需要重新分析 |
| 多次更新 | 新 BCE 必须与主 loss 在同一步前累积。先更新辅助项再复用旧 rollout，不能继续使用原 on-policy 解释 |
| stop-gradient | Q/基线/标签作为固定估计目标；不能让其通过 actor loss 额外反传 |
| 分叉独立性 | 条件于同一 $h$，fork 随机数需与主路线后续采样分离；不能按最终成功或实际 crop 筛选状态 |
| 动作与终止 | score 必须包含相关选择及终止支持；只给实际 crop 子集扣基线，一般不再是零均值控制变量 |
| 现有测试覆盖 | DB 测试调用了生产 advantage 函数，但最后累积的是玩具树上的 score×advantage；没有经过真实 token mean、完整 PPO loss、actor 和 optimizer，不能替代生产 loss 的无偏性证明 |

此外，公开 DB fork 的判分只检查解析到的 answer、字母与截断，没有执行完整生产 `check_step`。因此源码注释中的 “same rule” 不能作为已经对齐的证明。[DB 测试](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/pipeline/grpo/v51_db/test_db_credit.py)；[离线 fork](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/pipeline/eval/db/offline_forks.py)；[实际 loss](https://github.com/hiyouga/EasyR1/blob/602a12820f69cf45ea98de08a63b3509313a7d02/verl/trainer/core_algos.py)

这些限制不是说 baseline 方法不可用，而是说**“生产期望梯度不变”仍需针对实际损失证明，不能以简化测试通过代替。**

## 4.3 fc/zero 与新方案的实质区别

公开 brief 报告正在运行的是：

$$
c_{\rm fc}
=
\hat v(s_{t+1})-\hat v(s_t)-\lambda,
\qquad
c_{\rm zero}=-\lambda,
$$

并用这些系数**替换裁图轮 token 优势**。这是仍在运行的独立实验，不应称作已经废弃，也不能把它当作新 BCE 的实验结果。内部运行代码和预注册未公开，我只能核对公开转述，无法独立确认当前状态、精确 mask 和结果。

### fc 不自动获得势能塑形的保证

标准势能塑形把

$$
F_t=\gamma\Phi(s_{t+1})-\Phi(s_t)
$$

加到每个相关转移的 reward，并正确构造 return。固定 $\Phi$ 时：

$$
\sum_{t=0}^{T-1}\gamma^tF_t
=
-\Phi(s_0)+\gamma^T\Phi(s_T).
$$

若终止边界为零，新增项对固定初始状态是常数，因而不改变策略排序。这是势能塑形的核心条件。Ng 等作者页面确认其研究对象是这类保策略变换；原 PDF 的正文解析出现乱码，这里的有限时域结论是直接由上式推导。[Andrew Ng](https://wordpress.andrewng.org/index.php/publication/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/?utm_source=chatgpt.com)

fc 的公开定义则有三处不同：

- 它替换 crop-turn advantage，不是把 shaping 加入全轨迹 reward-to-go；
- $\Phi$ 是 answer-now 成功率，不是允许后续探索的 $V^\pi$；
- $-\lambda$ 对失败 crop 也存在，与 $Y(1-\lambda N)$ 的正确性门控成本不同。

最简单的一次 crop 后作答情形中：

$$
\text{导师路线收益差}
=(1-\lambda)q_c-q_a,
$$

$$
c_{\rm fc}=q_c-q_a-\lambda.
$$

两者相差：

$$
c_{\rm fc}-\text{路线收益差}
=-\lambda(1-q_c).
$$

所以 fc 对失败风险另有一项惩罚。

也要保留一个正确的例外：如果使用真正的 $V^\pi$、正确的即时 reward，并对完整动作支持使用 TD error，那么“用价值差作为 actor 优势”可以是标准 actor–critic。**不能仅因它替换 advantage 就否定它；当前缺的是这些条件，而不是一个新名称。**

---

# 五、与原任务收益的精确联系，以及 K=4 的关键风险

## 5.1 精确 Q 时：同一局部最优动作，不同梯度

固定状态 $h$、续跑策略和两条合法路线，令：

$$
\delta=Q_A-Q_C,\qquad p=\sigma(z).
$$

若以 $p$ 选择 answer，则局部期望收益为：

$$
J_h(p)=pQ_A+(1-p)Q_C=Q_C+p\delta.
$$

直接最大化收益的梯度：

$$
\frac{\partial(-J_h)}{\partial z}
=-\delta p(1-p).
$$

精确 Q 加权 BCE 的梯度：

$$
\frac{\partial\ell_{\rm BCE}}{\partial z}
=
\begin{cases}
-\delta(1-p),&\delta>0,\\
-\delta p,&\delta<0.
\end{cases}
$$

两者在独立、无约束的单状态问题中偏好同一个最优动作，但 BCE 对“当前很少选、分叉却认为更好”的动作更新更强。

这正是它可能有用的原因：**它不依赖那条稀少路线先在主 RLOO 的 16 条样本中频繁出现，便能直接监督其完整标签。**

但总体保证还缺少：

- 准确的 Q；
- 固定状态分布与续跑策略；
- 可以逐状态独立调整的函数类；
- 与部署动作概率一致的标签概率；
- 不相互干扰的共享参数更新。

当前共享 Transformer、有限 K、同步 RLOO、caption 分布变化和 optimizer 非线性都不满足这些理想条件。

## 5.2 K=4 不只是给精确 BCE 加了零均值噪声

以下是本次推导，不是仓库实验结果。

固定 $h,p$，设：

$$
D=\widehat Q_A-\widehat Q_C,\qquad
\mathbb E D=\delta,\qquad
\eta=\mathbb E|D|.
$$

导师损失的平均 log-odds 梯度为：

$$
\begin{aligned}
\mathbb E[g]
&=\mathbb E\!\left[|D|\bigl(p-\mathbf1[D>0]\bigr)\right]\\
&=\boxed{\eta\left(p-\frac12\right)-\frac{\delta}{2}}.
\end{aligned}
$$

因此，逐状态的期望损失最优概率为：

$$
\boxed{
p_K^*=\frac12+\frac{\delta}{2\eta}
}
\qquad(\eta>0).
$$

更明确的分解是：

$$
\boxed{
\mathbb E[\ell_K]
=
|\delta|\,\mathrm{BCE}(\mathbf1[\delta>0],p)
+
\frac{\eta-|\delta|}{2}
\bigl[-\log p-\log(1-p)\bigr].
}
$$

由于 $\eta\ge|\delta|$，第二项非负。它是**均匀目标交叉熵项**，不能直接称作 Shannon entropy bonus。

这带来四个结论：

1. Q 均值无偏，不意味着经过符号和绝对值后的 BCE 梯度无偏。
2. $|D|$ 是采样得到的收益差幅度，**不是置信度**。
3. 当真实差很小而采样波动很大时，回到接近各半的作用可能占主导。
4. 理想总体的分类方向仍可正确——$\delta>0$ 时 $p_K^*>.5$——但随机执行策略的最优概率和训练方向已经改变。

作为不增加训练臂的数学参照，可以考虑：

$$
\ell_{\rm EU}=-2pD.
$$

它满足：

$$
\mathbb E[\partial_z\ell_{\rm EU}]
=-2\delta p(1-p).
$$

在 $p=.5$ 处，EU 与 BCE 的逐样本梯度恰好相同，均为 $-D/2$。EU 没有上述额外项，但在概率饱和处梯度更弱。这是一个明确的权衡；本轮不把它再扩成第三条训练线。

## 5.3 两个精确反例

以下取 $K=4$、两分支独立、crop 恰好一次：

$$
B_A\sim{\rm Bin}(4,q_A),\quad
B_C\sim{\rm Bin}(4,q_C),
$$

$$
D=B_A/4-.99B_C/4.
$$

我实际枚举了全部 25 种成功次数组合，数值没有 Monte Carlo 误差。

| 条件 | 真实收益差 $\delta$ | $\mathbb E[g_{\rm BCE}]$ | 梯度下降的作用 |
|---|---:|---:|---|
| $q_A=.50,q_C=.51,p=.01$ | $-.0049$：crop 略好 | $-.13153346$ | **提高较差的 answer 概率** |
| $q_A=q_C=.50,p=.90$ | $+.005$：answer 略好 | $+.106875$ | **降低较好的 answer 概率** |

第一例尤其贴近目前几乎总 crop 的状态。它的：

$$
p_K^*\approx .49104,
$$

而直接 EU 的平均梯度是 $+.00009702$，继续偏向 crop。

所以，如果新方法显著提高直答比例，至少有两个竞争解释：

- 学到了哪些状态的直答路线确实更好；
- 有限 K 的 BCE 把原先饱和的动作概率拉了回来。

不能用动作比例本身区分。

---

# 六、四种情况如何处理，以及它们不能证明什么

先按导师原式手算。设 $p=.5$、$\lambda=.01$，crop 每条恰好一次。下面只是**观测到的四条样本结果**，不是对客观工具贡献的命名。

| answer / crop 成功数 | $\widehat Q_A$ | $\widehat Q_C$ | $D$ | 辅助梯度 $\partial_z\ell$ | 辅助作用 |
|---|---:|---:|---:|---:|---|
| 0/4，4/4 | 0 | .99 | −.99 | +.495 | 强烈偏向 crop |
| 4/4，4/4 | 1 | .99 | +.01 | −.005 | 轻微偏向 answer |
| 0/4，0/4 | 0 | 0 | 0 | 0 | 不增加动作监督 |
| 4/4，0/4 | 1 | 0 | +1 | −.5 | 强烈偏向 answer |
| 3/4，4/4 | .75 | .99 | −.24 | +.12 | 偏向 crop |

这里的方向只指辅助项，**不保证与 RLOO 合并后的参数更新方向完全一致**。

### 冗余工具

两边都答对时，成本让辅助项轻微偏向 answer。但原 RLOO 仍会把结果优势送到实际裁图轨迹的 assistant tokens。

因此，本方案不能描述成“冗余工具从此不再拿到正确功劳”。准确表述是：

> 保留结果学习，同时用路线比较在动作标签上增加校正。

### 必要但失败的探索

全零只意味着没有额外辅助信号。原 RLOO 仍可能对该失败轨迹产生负优势。

例如真实成功率为 $q_A=.01,q_C=.10$，crop 净收益明显更高，但两边 K4 全失败的概率为：

$$
(.99)^4(.9)^4\approx .63025.
$$

约 63% 的这种采样批次都不给辅助信号。**不额外处罚失败探索，不等于已经保护它，更不等于识别了其必要性。**

### 延迟 recovery：全续跑 Q 的一个实质优点

设第一次 crop 前后 answer-now 值都为零，但再 crop 一次可以稳定答对：

$$
Q_A=0,\qquad Q_C=1-2(.01)=.98.
$$

首个 crop 的即时 fc 系数可能为 $-.01$，而新路线比较得到：

$$
D=-.98,
$$

会支持先探索再恢复。这说明全续跑 Q 能包含延迟价值；它仍不证明第一次返回的像素本身有因果贡献。

### caption 已经走偏：一个结构性反例

即使原始图已经足够，模型也可能先写出一段遗漏关键信息、预先认定需要放大的 caption。若固定该 $h$ 后：

$$
Q_A(h)\ll Q_C(h),
$$

新损失就会正确地支持“从这个已形成的状态继续 crop”，却未必发现“换一种初始观察和 caption 后可以不 crop”。

所以**首次标签监督不是全部停止与推理效率问题的解法**。

### 同对、同错没有单次因果含义

两个动作各成功一次，可能对应 $0.55$ 与 $0.95$ 的真实成功率；各失败一次，也可能对应 $0.01$ 与 $0.4$。独立分叉估计的是条件期望差，并不恢复逐样本潜在结果的对应关系。

此外，crop 分支同时增加视觉信息和推理机会。这里接受这种混合，因为我们优化的是可部署路线；若要声称“像素信息导致提升”，还需要另一类干预，而本轮不增加它。

---

# 七、最小伪代码与 CPU 检查契约

## 7.1 伪代码

```python
# 固定：lambda = 0.01, beta = 1/256, G = 16, K = 4
# 第一轮 WORD 仅作意向；main / fork / eval 共用该协议
# aux loss 按预定状态数平均，main loss 保持原 token mean

for update in range(20):
    theta_old = current_policy_snapshot()
    questions = next_12_questions_in_fixed_order()
    slots = preselect_one_slot_per_question(seed_namespace="state")

    main_rollouts = rollout(
        theta_old, questions, n=16,
        protocol=P0, rng_namespace="main"
    )

    rewards = strict_reward(main_rollouts, lambda_=0.01)
    advantages = rewards - leave_one_out_mean(rewards)

    aux_records = []
    for q, slot in zip(questions, slots):
        h = exact_pre_action_snapshot(main_rollouts[q, slot])

        if not valid_prefix_boundary(h):
            aux_records.append(zero_aux_record())  # 不缩小分母
            continue

        returns = {}
        for action in ("answer", "crop"):
            returns[action] = []
            for k in range(4):
                tau = continue_from_clone(
                    theta_old,
                    h,
                    forced_complete_open_tag=tag[action],
                    remaining_budget=h.remaining_budget,
                    protocol=P0,
                    rng_namespace=("fork", update, q.uid, action, k),
                )
                returns[action].append(strict_reward(tau, lambda_=0.01))

        delta = stop_gradient(
            mean(returns["answer"]) - mean(returns["crop"])
        )
        aux_records.append((h, delta))

    L_main = production_rloo_loss(main_rollouts, advantages)

    L_aux_sum = 0
    for h, delta in usable(aux_records):
        s_answer = full_tag_logprob(theta, h, ANSWER_TAG)
        s_crop   = full_tag_logprob(theta, h, CROP_TAG)
        z = s_answer - s_crop
        L_aux_sum += relu(delta) * softplus(-z)
        L_aux_sum += relu(-delta) * softplus(z)

    L_aux = L_aux_sum / 12
    optimizer_step(L_main + beta * L_aux)  # 同一次更新
```

额外分叉只提供 Q 标签，不进入主 RLOO 的 16 条 on-policy 轨迹组，也不把强制标签冒充学生自然采样的 token。

## 7.2 必须先通过的检查

这些检查应直接调用新实现实际使用的 loss helper、mask/reduction 和协议函数。它们证明实现符合设计，**不证明研究有效**。

| 检查 | 必须满足的结果 |
|---|---|
| 首轮合法动作 | 同一个合法 caption 前缀，两种强制开标签均可进入对应动作；后续仍执行正常几何、来源和预算检查 |
| 原始边界 | h 来自实际 token ledger；禁止解析器归一化文本后重新 tokenize，再声称是原 h |
| 完整标签概率 | 使用真实 tokenizer；全序列 log-prob 与逐 token 求和一致；共享首 token 不能代替全标签 |
| 符号与平局 | $D>0$ 提高 answer 相对偏好；$D<0$ 相反；$D=0$ 严格零梯度 |
| 小成本尺度 | 所有 $D$ 乘 .01，辅助梯度也精确乘 .01；不能被归一化抵消 |
| mask | 直接计分仅覆盖完整开标签；分叉正文不进 aux；不能误把“前缀无 loss”实现成切断前缀表示梯度 |
| reduction | 改 microbatch 切分、padding、分布式分片后，固定全局状态均值保持一致 |
| stop-gradient | Q 与 D 无 actor 梯度；teacher-forced 标签 scorer 有梯度 |
| RNG 与真值隔离 | fork 不推进主 rollout 的随机数流；改变 gold 只改变评分，不改变模型输入 |
| 预算与终止 | 分叉继承剩余预算；训练与评测对 invalid、截断及终答使用同一语义 |

还需把上面的 25 种离散组合送入**实际辅助 loss 函数**，重现反例梯度。这里检验的是非线性变换是否按预期实现，不应再将它包装成完整生产策略梯度的验证。

当前 actor 的字段白名单也需要接线：新 prefix/pair 数据若未加入独立 aux batch 或白名单，会在训练前被丢弃。当前公开评测则记录 `check_errors` 却不立即停止，并同时保留宽松 `ok` 与严格 `ok_train_rule`。新 MVE 必须以统一严格执行语义为主，不能让格式冲突通过宽松判分掩盖。[eval_v4.py](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/pipeline/eval/eval_v4.py)

### 旧 m=4 门槛应如何处理

旧离线程序确实计算的是 $\mathbb E[c^2]$，而实际梯度形如：

$$
g=\frac{cS_\theta}{M}.
$$

其方差取决于：

$$
\mathbb E\!\left[\frac{c^2S_\theta S_\theta^\top}{M^2}\right]
-\mathbb E[g]\mathbb E[g]^\top,
$$

不等于 $\mathbb E[c^2]$。

自然作答也随机时，单独的 `.777` 一致率不能识别 prefill 偏差；即使两次答案来自完全相同分布，一致率也只有 $\sum_y p(y)^2$。需要同状态重复作答的噪声参照。

因此，旧门槛未过应保留原结论，不能改阈值补过关；但它既没有检验新 BCE，也不支持“增加分叉就一定修好”。

---

# 八、唯一最小 MVE：一条主线，一个必要对照

## 8.1 为什么必须有一个新的协议匹配对照

新 brief 希望复用 r1，避免重跑基线。我同意不重跑 165 步基线，但公开代码中的 WORD 锁定使协议修改成为前提。

**历史 r1 可以提供共同权重起点和背景，不能单独隔离“协议变化”与“新增 BCE”。**

因此唯一新增训练对照为：

- **主臂：新首轮协议 + 原 RLOO + 上述 BCE。**
- **对照：同一新首轮协议 + 原 RLOO，$\beta=0$。**

不再新增 EU、打乱标签、不同 K、不同 $\lambda$、不同 mask 的训练矩阵。

## 8.2 共同起点与更新范围

建议从已有 **r1-step40 权重**做局部探针，每臂最多 **20 次更新**，使用同样的主训练题序、seed 42、$G=16$、学习率配置、KL/DS 设置和原有全局梯度裁剪规则。

这检验的是高裁图倾向阶段的局部可改性，不代替“从 SFT-A 完整跑 165 步”的结论。

公开脚本设置 `save_model_only=true`，因此不能假定已有 optimizer/scheduler/RNG 状态可精确恢复。若只有权重，两臂必须同样重置，并明确记录为**共同权重上的局部重启**。[启动脚本](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/pipeline/grpo/scripts/grpo_rloo.sh)

**β=0 对照不生成随后丢弃的训练分叉。** 那样只会烧掉相同资源，并没有让控制策略消费同样的信息，不能识别 BCE 形式的独特贡献。

这意味着本 MVE 比较的是：

> 同协议、同主 rollout 数、同更新次数下，增加“分叉估值＋动作监督”整个方案的效果和额外成本。

它不宣称等算力优越，也不宣称 BCE 优于其他使用同样 Q 信息的损失。

## 8.3 三层证据

### 第一层：标签是否真的进入动作选择

报告辅助 loss、$z$、$p$、规范标签总质量 $m$、非法率，以及不同 WORD/收益差区间的变化。

这些只是实现和拟合诊断，不能当研究成功。

### 第二层：独立状态上的选择收益是否改善

预先固定 **64 个不进入训练的共同 $h$**，由共同起点、同一新协议产生；每个状态 answer/crop 各 K4，用独立随机数得到 $\widehat\delta_{\rm audit}$。

两臂训练结束后，仅在这些相同 $h$ 上打分：

$$
\boxed{
\widehat U_{\rm choice}
=
\frac1{64}\sum_h
\left(p_{\rm main}(h)-p_{\rm ctrl}(h)\right)
\widehat\delta_{\rm audit}(h)
}
$$

按题目聚类计算区间。

它估计的是：**固定参考续跑策略下，两动作归一化选择概率的变化带来的局部收益差**。它不是新学生整条策略收益的估计；后续能力、状态分布和非法输出仍需第三层测量。

### 第三层：真实准确率、成本与恢复

在同 64 题上，两模型 clean/E2 各采样两次，报告：

- 严格准确率差及配对区间；
- $Y(1-.01N)$、crop 数和生成 token 差；
- 实际处理 token、延迟；
- E2 中实际遭遇错框的比例。

E2 整体变好可能只是更少进入 crop，不能独立证明 recovery 保持。为此，在这 64 题中**预先指定 16 题**，重放共同参考分支的首个 crop，按原 E2 规则构造同一错框后状态；两模型各续跑两次。它只增加 64 条 continuation，却能避免按各模型是否主动 crop 选择恢复样本。若参考分支无法形成有效 crop，就报告缺失与覆盖率，不按最终成败补选。

这不是恢复能力的高功效确认实验，但能筛出明显的共享参数伤害。

## 8.4 预算上限

| 项目 | 数量 |
|---|---:|
| 两臂主 rollout | $2\times20\times12\times16=7,680$ |
| 仅主臂训练分叉 | $20\times12\times8=1,920$ |
| 64 个共同状态的独立估值 | $64\times8=512$ |
| 两臂 clean/E2，各题各两次 | $64\times2\times2\times2=512$ |
| 16 个相同错框后状态的恢复续跑 | $16\times2\times2=64$ |
| **完整/分支轨迹合计** | **10,688** |

另计 64 个参考前缀、成对完整标签 scoring、视觉预填和反向计算。

**建议硬上限：合计 8 GPU 小时、400 万生成 token、5,000 万处理 token，任一先到即停止。** 这是设计预算，不是已经测得的运行时间。

第一批 12 个状态的计时计入该预算。若按实际实现预测无法完成预定范围，应在更新前报告预算不满足；不能临时缩 K、缩 crop cap 或截短较贵的分支来凑完实验。

报告必须区分：

- 主 rollout；
- answer forks；
- crop-continue forks；
- prefix/image prefill；
- 标签 forward/backward；
- 评测。

若最终确有推理节约，还可报告：

$$
\text{训练额外开销的回收题量}
=
\frac{\Delta C_{\rm train}}
{C_{\rm infer,ctrl}-C_{\rm infer,main}},
$$

使用同硬件和计量单位。不能只报“推理少了多少 crop”，遗漏训练时买这些标签的成本。

## 8.5 预先固定的停止与反证规则

| 观测 | 结论及行动 |
|---|---|
| 同 h 两合法动作、边界、mask 或 reduction 检查失败 | 不启动更新；实现尚未对应指定算法 |
| CE 降、直答增，但独立 $U_{\rm choice}$ 没有改善 | 不支持“学会了更好的选路”；不以比例变化宣布成功 |
| $U_{\rm choice}$ 的区间明确为负 | 反对当前局部动作改进解释；停止扩大该配置 |
| 合法输出显著恶化，或动作趋于单一且持续忽略反向收益证据 | 判为失败；单一动作比例本身不是目标或独立判罪标准 |
| clean 或共同错框状态出现明确伤害 | 放弃当前配置；不能用少 crop 抵消未声明的准确率损失 |
| 区间跨零 | 结论不确定，预算到点停止；不等于机制已被否定，也不等于准确率不降 |
| 预算越界 | 工程成本未满足；不改原门槛补过关 |

准确率非劣界限必须预先声明。例如可把 **2 个百分点**作为这次探针的伤害筛查界限，但这只是建议，不是替导师接受永久损失。若要正式声称“不劣超过 2 点”，需要区间下界达到该标准；64 题通常不足以保证这种精度。

E1 已被反复开发使用，40 题疑似重叠也必须处理。上述 64 题应按题目、图像及近重复关系排除 SFT/RL 和已用于设计的病例。若找不到真正隔离的集合，只能把结果标为开发诊断，不能改名为确认实验。

---

# 九、原文查新、与旧失败的区别，以及执行交接

## 9.1 最近的理论先例是 DPI/CAPI

直接相关的经典方法已经包含：

1. 采样状态；
2. 每个状态、每个动作做多次 rollout，首动作固定、后续执行当前策略；
3. 用动作 Q 差衡量选错动作的代价；
4. 训练动作分类器。

DPI 的定义与算法框架明确采用这种 rollout-based policy improvement；CAPI 进一步写成 action-gap-weighted classification，并讨论可微分类代理。**导师公式是这个思路在 VLM 动作字符串上的具体实现，不需要被包装成新的 advantage 原理。** 这些原文也不直接给当前共享 Transformer、有限 K、BCE 与 RLOO 联合更新提供单调改进保证。[jmlr.org](https://jmlr.org/papers/volume17/10-364/10-364.pdf?utm_source=chatgpt.com)

## 9.2 K11/K19/K36 各自覆盖哪一层

| 卡与已核原文 | 真正覆盖的部件 | 与本方案的边界 |
|---|---|---|
| K11 / VisionThink | 按组内工具相关表现构造条件惩罚，经 GRPO 学习工具使用 | 其比例来自两类正确数量，不是同一真实 caption 后的双动作 Q；没有这里的标签 BCE |
| K11 / AdaTooler-V | 用参考模型的 with/without-tool 表现差构造工具 reward | 题目级离线信号与 trajectory reward，不是当前学生局部动作分类 |
| K19 / MM-PRM | 从推理前缀 MC 续跑，以成功率形成监督 | 训练状态评分器；可达成功率不等于当前步骤的事实正确性 |
| K36 / RLHF-V | 共享回答主体、局部修正，使用加权偏好学习 | 有 reference 和 DPO 结构；未改 token 仍有权重。这里仅是局部监督类比，不能称为 DPO |

这些区别来自原文方法段，而不只是卡片摘要。[arxiv.org](https://arxiv.org/html/2507.13348v1?utm_source=chatgpt.com)

相关 parent/subpattern 对本轮最有用的要求是：固定比较对象、识别竞争解释、检查协议混杂、验证信号是否到达预定位置。它们提供研究组织方法，**不提供新颖性证明，也不要求把多个模块拼进算法**。

## 9.3 与旧 $s\cdot dV$、CounterCredit、TACO 的区别

| 方法 | 测量与注入方式 | 本方案的实质差别 |
|---|---|---|
| 旧首轮 $s\cdot dV$ | 冻结 SFT-A、题目级有/无首轮工具分支；caption 不同；正确率差加到首轮整段优势 | 新方案固定实际 caption 后 h、用当前策略完整续跑净收益、只新增标签分类 loss |
| CounterCredit | 实际工具分支与 answer-now、随机像素返回比较 gold-answer utility，再构造价格信号 | 新方案测完整动作路线回报，不认证视觉像素贡献，也不使用其 rent/cashback |
| TACO | 工具前后 greedy answer probes，加结果/过程优势路由 | 新方案用多次完整续跑均值直接训练动作概率，不依赖单次前后对错来判责任 |

旧 $s\cdot dV$ 实验在 step160 增加约 1.72 次 crop，准确率差区间跨零；这个失败应保留。新方案改变了估值状态、当前策略、收益定义和接收位置，因而不是完全相同实验，但这些区别**还不是它一定会成功的证据**。[旧 dV 实验原文](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/docs/29_twobranch_dv_grpo_20261020.md)

CounterCredit 与 TACO 的上述机制分别见其原文方法部分；我没有把其“需要、使用、责任”等命名直接接受为本项目已经识别的因果量。[arxiv.org](https://arxiv.org/html/2609.22910v1?utm_source=chatgpt.com)

**最保守的新颖性判断：监督原理已有明确先例；本项目的潜在增量是这个具体接口、有限预算估值与部署效果。** 现在还不能宣布它是新的信用分配原理。

## 9.4 实际已读与未读

**本轮本人已读：**

- 三个新入口全文。
- 合并报告 `docs/30_rl_credit_report_20261022.md`、旧失败实验 `docs/29_twobranch_dv_grpo_20261020.md`。
- K11、K19、K36、K17、K18、K22、K25、K46，合计 **8 张机制卡全文**。
- 14 父模式、19 子模式的概览；5 个相关父模式全文：`diagnose_then_target`、`repair_degenerate_signal`、`manufacture_contrastive_pairs`、`reward_explicit_intermediates`、`decompose_and_reward_parts`。
- C00、C02、C04、C05、C07、C13、C16，合计 **7 个子模式全文**。
- 正式 RLOO 启动链、cropfloor rollout/reward/token 契约、协议校验、DB 核心与测试、offline forks、评测关键路径。
- 上游 EasyR1 固定版本 actor 全文及 RLOO/reduction/PPO loss、rollout logits 配置相关段落。
- u17/u39 文字与实际图片。
- DPI/CAPI、VisionThink、AdaTooler-V、MM-PRM、RLHF-V、CounterCredit、TACO 的相关原文方法段。

**未读或未独立核验：**

- 其余 **50 张机制卡、9 个父模式、12 个子模式全文**；看过目录和概览不算全文阅读。
- K11 的 GeoEyes、K19 的其余全部论文、K36 的另一篇原文等，不能据卡片补写结论。
- 旧公开 idea card 只读了核心方法与门槛等相关段落，未把全部领域附录计作精读；内部旧 card/JSON 未公开。
- `lark/Conversation.md` 原始内部对话；本轮读到的是新包公开摘录。
- fc/zero 内部运行代码、预注册、实时状态和结果；公开 tree 没有这些材料。
- 实际 checkpoint、完整 tokenizer 文件、训练数据去重结果以及新 BCE 的生产运行。
- `SOURCE_MANIFEST` 已解析范围与相关条目，未逐条验收全部历史阅读记录。
- Ng 原 PDF 的正文解析失败；只核实了作者页面，相关有限时域结论在上文自行推导。

## 9.5 可直接交给执行者的最小工作包

只需要四项交付，完成后再判断是否进入上述小训练：

1. **协议与状态契约**：首轮 WORD 意向语义；统一 main/fork/eval；真实 token/image snapshot；模型、tokenizer、SYSTEM 和数据版本标识。
2. **一个分叉函数与一个完整标签 loss helper**：K4+K4、继承预算、严格评分、标签求和、独立状态均值、同一次 optimizer step。
3. **CPU 检查记录**：包含真实 tokenizer 边界、实际 helper 梯度、reduction、协议一致性及本文有限 K 反例。
4. **一份冻结的 MVE 配置和结果账本**：两臂、20 更新、$\lambda=.01$、$\beta=1/256$、固定题序与独立 RNG、64 个审计状态、预算及停止规则；保留每条 fork 的 $Y,N,R$，不能只保存平均 Q。

### 给导师的简短解释

我们把信用问题收窄为“同一 caption 后该选哪条路线”。原 RLOO 保留答题学习，额外分叉估计现在答与先裁图再继续的净收益，只把比较结果用于完整动作标签监督。这属于已有的 rollout 策略改进思路。公开代码需要先解除首轮 WORD 对动作的提前锁定；K=4 加权 BCE 还会产生使动作概率回到接近各半的作用。因此，先用一个短程、同协议的 RLOO 对照，检查动作变化是否对齐独立收益差，并同时看准确率、成本和共同错框状态下的恢复。直答比例上升或 CE 下降，都不能单独算成功。
