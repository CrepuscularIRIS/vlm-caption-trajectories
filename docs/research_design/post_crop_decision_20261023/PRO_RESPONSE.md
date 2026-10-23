# Pro 完整回复：首裁后的可停止性

2026-10-23；网页模式 Pro，已观察到 `Response complete`。这是模型设计咨询，原文阅读范围是 Pro 自报，效果未经实验验证。仅将数学分隔符调整为 GitHub Markdown；原文哈希见 [RESPONSE_RECEIPT.json](RESPONSE_RECEIPT.json)。Codex 的结论与待补项见 [READOUT.md](READOUT.md)。

---

**我接受 Stage 0 的停止结果：不进入首轮小头 Stage 1，也不把同一个小头移到后裁位置继续试。当前建议仍是暂不训练。**

这轮可以形成一张有明确研究价值、也允许被否定的主卡：

> **首裁回执之后，原模型是否还存在可辨认的停止机会？这些机会能否在保留必要取证与 recovery 的条件下，最终变成 base model 自己的行为？**

这里有三个尚未接上的环节：**有停止空间、当时可见的信息能识别它、原模型能学进去而不破坏作答。** 现有材料没有证明其中任何一环已经在后裁状态成立。

我实际读取了新入口提交 **`63c28ef843f1fa19f296eef0cf20a64cc8ebfa61`**，以及 Stage 0 的完整提交 **`5399cba8f4f8bb8d365b457a412409774dddaed2`**。下面按新 README 的八项问题组织。此次只做公开材料审查和公开数字的算术复核，没有新采样或训练。[本轮任务入口](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/63c28ef843f1fa19f296eef0cf20a64cc8ebfa61/docs/research_design/post_crop_decision_20261023/README.md)

---

# 一、先用人话解释导师的建模顺序，以及此前选错了什么

导师的思路可以理解为：

> 先找出一个具体时刻，问模型此时究竟能看到什么、能合法做什么。然后比较两条可执行的后续路线，看看选择不同路线到底改变了什么。只有存在值得学习的差别，才决定把信号给哪个输出、用什么损失去学。

这与先寻找一个“更好的 credit 公式”有很大区别。公式应当服务于一个已经定义清楚、能够测量的行为问题。

## 1.1 四个层次应当继续分开

| 层次 | 本项目里回答的问题 | 它本身不能证明什么 |
|---|---|---|
| **任务目标／reward** | 我们愿意怎样交换正确率与成本？ | 奖励提高不自动等于准确率不降 |
| **advantage／baseline** | 相对于什么参照、以什么尺度更新？ | 方差降低不自动产生新的停止机会 |
| **信用路由** | 哪些 token 接收哪个信号？ | 更细的 mask 不自动识别工具的因果贡献 |
| **动作监督** | 直接怎样改变停止／继续的选择？ | 标签拟合变好不自动意味着自然行为改善 |

合法状态基线可以减少策略梯度估计的方差；在相应条件成立时，它不会改变两个动作谁更值得选。有限训练中它可能改善学习效率，但“保持期望梯度不变”不能成为这项研究的硬目标。导师希望改变的是模型的实际取舍。

同时，`LESSONS.md` 中“只在裁图动作上改信号通常改变目标”的说法还可以更精确：它至少改变了更新估计或更新向量场；**任意一种 token 改写未必恰好对应某个清楚的新标量目标。** 动作相关的 advantage 本身完全合法，问题在于把未经补偿的动作相关项冒称为零期望 baseline。[工程与建模教训](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/63c28ef843f1fa19f296eef0cf20a64cc8ebfa61/docs/research_design/post_crop_decision_20261023/LESSONS.md)

## 1.2 “正确且自信时 BCE 梯度小”仍然不是已经查明的缺陷

令 $D=Q_S-Q_C$，$p$ 是停止概率分数。原加权 BCE 的标量梯度是：

$$
\frac{\partial L}{\partial z}
=|D|\bigl(p-\mathbf1[D>0]\bigr).
$$

如果 $D<0$，应继续裁图，而模型已经 $p\approx0$，小梯度是正常现象：**这个状态的动作已经选对，没有必要再大力推它。**

真正需要解释的是：为什么其他状态上的更新，会把这些本来选对的状态也推向停止？这涉及共享参数、RLOO 与辅助项的合成、优化器、更新幅度和标签噪声。全局 logit 上移是一种观测结果，不能单凭它确定是哪一种机制。

有限 $K$ 确实可能改变 BCE 的平均监督。若 $\widehat D$ 是带噪估计，则：

$$
\mathbb E\!\left[\frac{\partial L}{\partial z}\middle|h\right]
=
\left(p-\tfrac12\right)\mathbb E[|\widehat D|\mid h]
-\tfrac12D(h).
$$

因此，接近零的真实收益差也可能产生向中间概率收缩的压力。但这只是一个可检验机制，**不是对旧实验全局漂移的完整归因**。三轮失败报告中“原因是饱和”的措辞应保留为假说，不作为下一轮直接换 loss 的依据。[三轮实验总报告，第 7 节](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/63c28ef843f1fa19f296eef0cf20a64cc8ebfa61/docs/30_rl_credit_report_20261022.md)

## 1.3 Stage 0 应保留的结论与需要收紧的解释

**已读事实与我的复核：**

| 项目 | 公开结果 | 合理解释 |
|---|---:|---|
| 首轮审计状态 | 64 个，参考模型 r1@40，每路 K4 | 当前首轮开发样本 |
| 两半 $D$ 的相关 | .705，CI [.526, .831] | 存在较强的跨状态差异 |
| “信号方差占比” | .826 | 依赖条件独立、零均值噪声等假设的总体方差分解 |
| 同表逐状态选择超过最佳阈值 | .008164 | 当前表上的新增空间较小 |
| 跨半差值 | −.01129，CI [−.03836, .01195] | 不支持继续投入原首轮小头 |
| 首轮 crop 分支均值 | 1.8086 次 | 有后续成本暴露，尚无冗余标签 |

这里有三项不能升级解释。

**第一，全局 $D$ 可靠不等于停止边界可靠。** 大量明显应裁图、$D$ 很负的状态，足以产生很高的总体相关；真正接近 $D=0$ 的少数状态仍可能难以区分。`40/43` 的符号一致率还排除了 21 个至少一半为零的状态，不能改写成全体标签准确率。

**第二，2+2 跨半比较不是新题泛化。** 它是在同一批状态上换续写样本；现有 bootstrap 又固定了已经选出的阈值，没有重做阈值选择全过程。它足以支持当前资源停止决定，不能证明总体上不存在更好的首轮策略。

**第三，K4 下“没有观察到准确率交换”有分辨率原因。** 两路准确率差的最小负步长是 $-.25$，而 $\lambda=.01,N\le8$ 的最大成本补偿不超过 .08。因此：

$$
\widehat{\Delta Y}<0
\quad\Longrightarrow\quad
\widehat D\le-.25+.08<0.
$$

所以这张表里没有“$D>0$ 但 observed accuracy 更差”，不能作为真实 Pareto 保证。[Stage 0 脚本与结果](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/tree/5399cba8f4f8bb8d365b457a412409774dddaed2/logs/dl/stage0)

**勘误核对成立：**五个 $D>0$ 中，参考 $z>14$ 的只有 `Math_492` 和 `Biology_60` 两个。第三个高 $z$ 状态 `Agriculture_22` 的 $D=0$。我从公开五条正值记录复算，正值总和除以 64 为 .0316796875，去掉上述两个高 $z$ 状态后的贡献为 .0081640625，与公开结果一致。[AUDIT.json](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/63c28ef843f1fa19f296eef0cf20a64cc8ebfa61/docs/research_design/post_crop_decision_20261023/AUDIT.json)

另外，256 条强制 crop 分支中有 95 条 $N\ge2$，首裁之后合计执行了：

$$
\sum (N-1)_+=210
$$

次裁图。**210 是发生过的后续调用数，不是可删除的冗余数。** 这些还是强制首裁分布，不能直接代表自然策略的后裁分布。

此前最重要的对象错误，是把“首轮是否开始看”“后续是否还要看”“已有证据如何用于作答”都放进了同一个信用分配叙事。现在应当收窄到一个明确决策点。

---

# 二、一张主卡：首裁回执后的停止／继续价值

## 主卡

| 字段 | 内容 |
|---|---|
| **名称** | 首裁回执后的可停止性 |
| **状态** | 待证问题卡；当前不进入训练 |
| **研究对象** | 已成功收到第一张裁图回执后，是否值得再进行一次取证并继续完成 |
| **决策时点** | 首裁回执结束后的 assistant-start；下一轮尚未生成任何 token |
| **可见信息** | 问题、原图、实际显示的首裁图、全部历史、source/bbox/回执、剩余预算；不包含未来答案或正确性 |
| **停止选项 $S$** | 进入合法回答 WORD 路线，让同一冻结模型完成 caption 与答案 |
| **继续选项 $C$** | 进入合法 grounding WORD 路线，执行首个新增 crop，随后由同一冻结模型自然完成，允许继续探索和 recovery |
| **主测量** | $D(h)=Q_S(h)-Q_C(h)$；另测相对原自然策略的实际收益空间 |
| **后续梯度接收者** | 原 LM 对完整 WORD 路径的相对概率；不增加推理时小头或 inspector |
| **预期行为** | 在停止价值较高的状态少看；在继续能提升成功率的状态保留整段探索机会 |
| **最小否证** | 合法测量后没有足够停止空间；或只有格式干预收益；或可见信息无法稳定识别；或内化后自然作答／recovery 受损 |
| **当前决定** | 先审计旧轨迹；必要时才讨论一次有硬预算的无训练分叉 Probe |

**范围刻意限定在第一张回执之后。** 它直接研究第二次 crop 是否值得发生。若结果为阴性，不能顺便宣布第三次、第四次裁图之后也没有停止空间；但本轮不同时铺开所有时间点。

## 2.1 这与 advantage、credit assignment 分别有什么关系？

在一个确实由 $S,C$ 两个选项构成的抽象策略中，设选择停止的概率为 $\eta$，则：

$$
V_\eta=\eta Q_S+(1-\eta)Q_C,
$$

$$
A(S)=(1-\eta)D,\qquad A(C)=-\eta D.
$$

所以 $D$ 可以构造动作 advantage，直接回答“这时往哪边更新”。

但它的测量对象是**两条后续策略路线的价值差**。它没有把最终成功唯一归因于某张 crop、某句 caption 或某个像素。继续路线同时改变新视觉输入、推理长度、选框以及 recovery；这些都是路线价值的一部分。

因此，本卡首先是停止决策问题；把它落实到 WORD 输出后，才涉及信用路由。它尚不是“真实工具贡献已经被识别”的卡。

## 2.2 实际匹配的 cards／patterns

我读了以下卡的全文，并回到相关原文核对了关键机制：

| 卡／pattern | 本卡采用什么 | 不采用什么 |
|---|---|---|
| **K19：Monte Carlo step labels** | 同一多模态前缀下，通过终答续写估计可达收益 | 不把成功可达性当成视觉事实正确性 |
| **K11：benefit-conditioned tool reward** | 根据工具路线的实际收益讨论是否鼓励 | 不把题目级“工具有用”复制成每一后续轮都该裁图；不默认提高 $\lambda$ |
| **K57：decouple answer authority** | 提醒我们区分取证与作答充分性 | 不采用外置 inspector 接管终止和答案，这不满足最终自主 base 的目标 |
| **diagnose_then_target → C07** | 先审计测量工具，再定位问题，再做针对性改变 | 不把一个通用损失接在描述性诊断后就称为机制成立 |
| **measure_example_utility → C11** | 借用成本核算、强简单对照的要求 | 其对象是训练数据选择，与后裁停止只有类比关系 |

主匹配是 **K19 的估值部件，加上 C07 的诊断顺序**。没有必要将三张机制卡拼成一个新系统。[K19](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/63c28ef843f1fa19f296eef0cf20a64cc8ebfa61/docs/research_design/decision_loss_20261022/mechanism_cards/K19-monte-carlo-step-labels.md) · [K11](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/63c28ef843f1fa19f296eef0cf20a64cc8ebfa61/docs/research_design/decision_loss_20261022/mechanism_cards/K11-benefit-conditioned-tool-reward.md) · [K57](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/63c28ef843f1fa19f296eef0cf20a64cc8ebfa61/docs/research_design/decision_loss_20261022/mechanism_cards/K57-decouple-answer-authority.md) · [C07](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/63c28ef843f1fa19f296eef0cf20a64cc8ebfa61/docs/research_design/decision_loss_20261022/patterns/sub/C07.md)

---

# 三、唯一的最小无训练 Probe：先核查旧轨迹，必要时才补两个后裁选项

## 3.1 当前先做的零 GPU 工作

继续固定 **SFT-A 与 r1@40 reference**，避免同时更换研究时点和 checkpoint。权重、tokenizer、processor、协议、采样配置都需要在执行清单中固定；公开报告中的名称不能替代实际 hash。

从已有自然完整轨迹中，按题／共享图像重复组预先确定一个 rollout 槽位。每条轨迹只取首裁回执后的一个状态。选择规则只依赖截至此处的历史。

**不得用以下条件选样：**

- 最后答错；
- 后面至少裁了两次；
- caption 出现了正确答案；
- 后来是否成功 recovery。

自然直答、首裁前 invalid、没有成功收到回执的轨迹，仍留在整体覆盖率的分母中。

这次扫描只回答三个问题：

1. **状态是否存在且能准确重建？**  
   是否有原 token IDs、实际图像／grid、环境状态、剩余预算？
2. **后续成本有多大？**  
   记录未来 crop、生成 token、反复输入的图像／文本 token、延迟和预算终止。
3. **即使机械地删除全部后续调用，节省上限是否值得研究？**  
   这个上限可以用来停止项目，但不能当作可实现节省。

公开 `dl_audit.py` 保存的是分叉终值、正确性、crop 数、奖励和生成 token，**没有保存每条分支内部的完整后裁 ledger**。现有 eval JSON 也不等于原生产 token ledger：它保存文本和解析结果，并在每轮重新套 chat template。仅拿终值表或文本重建，不能声称已经获得“完全相同的生产状态”。[dl_audit.py，第 121–136 行](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/5399cba8f4f8bb8d365b457a412409774dddaed2/pipeline/grpo/v52_dl/dl_audit.py#L121-L136) · [eval_v4.py，第 147–207 行](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/5399cba8f4f8bb8d365b457a412409774dddaed2/pipeline/eval/eval_v4.py#L147-L207)

**当前我没有这些私有完整日志，所以不能报告这一步已经通过。**

## 3.2 何时才需要新 fork？

只有同时满足以下条件，才值得讨论新增测量：

- 后续成本暴露足够；
- 合法共同状态可以准确重建；
- 已有缓存没有该状态的两个选项结果。

若保留主卡的 $D=Q_S-Q_C$，最低完整测量是：

> **每个状态新增一条 $S$、一条 $C$，各 K=1；复用原自然 suffix 作参照。**

只补停止分支更便宜，但测到的是：

$$
T(h)=Q_S(h)-V_{\mathrm{nat}}(h),
$$

而不是 $D(h)$。当前不能假设规范 WORD 分叉恰好等于自然策略的条件分布，所以不能用 $T$ 偷换 $D$。

增加 $C$ 的作用，是区分“停止优于继续取证”和“规范前缀整体改变了作答方式”。它补齐了主卡真正缺失的反事实。

## 3.3 建议的单次预算上限

以下是**供讨论的支出硬帽，不是已获授权的任务，也不是耗时预测**。

| 项目 | 上限 |
|---|---:|
| 独立题／图像组 | 64 |
| 每组新分叉 | $S,C$ 各 1 条 |
| 新增完整 suffix | 128 条 |
| 新生成 assistant token | 0.40M |
| processed input token，含图像 | 8M |
| GPU 支出 | 2 GPU 小时 |
| 终点分析 | 1 次，不因结果接近门槛而补 K／补题 |

建议预先在 64 个组中安排 **48 个 clean、16 个已有错框条件**，分别报告。它们不能被混成一个“总体准确率”；同题、同图、近重复必须归在同一组。若没有合格错框缓存，就明确 recovery 未测，不临时制造阳性病例。

沿当前每轮 640 token、最多 8 次 crop 的协议，第一裁之后：

- $S$ 最多一个回答轮；
- $C$ 最多还有 7 次 crop 和一个最终回答轮。

因此，新增 assistant token 的宽松上界是：

$$
64(640+8\times640)=368{,}640.
$$

完整 WORD 打分、图像预填、模型加载、失败调用及重试也计入预算。**不能把 $C$ 截成“再裁一次就必须回答”，也不能只给它 640 token 来匹配 $S$ 的计算量。** 那会删除长链 recovery，改变所测的 $Q_C$。

协议自身的 invalid、长度终止、context limit 按统一任务规则评分。研究支出硬帽导致未完成的分叉则属于缺失测量，不能伪装成模型答错；如果无法完成预定配对，就停止科学判定。

## 3.4 这个 Probe 的能力边界

K1 适合便宜地排除粗大失败，不适合给每个状态贴“该停／该裁”的训练标签。

例如两路成功率都约 .5、条件独立时，64 题 K1 的准确率差，仅续写噪声带来的近似 95% 半宽就是：

$$
1.96\sqrt{\frac{.5}{64}}\approx .173.
$$

即使换成 K4，仍约为 .087。实际方差可能不同，但这足以说明：**不能把这样的小 Probe 当成严格保准确率或边界标签可靠性的验收。**

如果导师要求这一次测量就证明严格不降准确率，那么该预算与主张不匹配，应当直接暂不采样。若接受它只是一次资源决策用的否证筛查，则可以在状态核查通过后讨论执行。

---

# 四、协议选择：共同前缀必须在下一轮任何输出之前

这是本卡能否成立的技术前提。

## 4.1 选择明确的 $h_0$

我选择：

> **首裁图像与回执完整加入上下文后，末尾的 assistant-start。**

此时下一轮的 WORD、caption、动作都尚未生成。

生产 adapter 已经有这个时间点：`receipt_chunk` 以 assistant-start 结束；新回执、图像和 grid 原子追加后，下一轮开始采样。v5.2i 的松绑只发生在 Step 1，Step 2 仍沿用 WORD 与 action kind 的约束。[caption_v52i.py](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/5399cba8f4f8bb8d365b457a412409774dddaed2/pipeline/grpo/v52_dl/caption_v52i.py) · [easyr1_caption.py，第 456–535 行](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/5399cba8f4f8bb8d365b457a412409774dddaed2/pipeline/grpo/v52_dl/easyr1_caption.py#L456-L535)

两个选项内部使用完整规范前缀：

```text
<think>
Step 2 | WORD
```

其中包含 WORD 后的换行：

$$
\mathcal W_S=\{\text{commit},\text{abstain}\},
$$

$$
\mathcal W_C=\{\text{locate},\text{re\_examine},\text{fallback}\}.
$$

然后让冻结模型生成 caption、动作内容及后续轨迹。`abstain` 仍必须合法输出 `UNCLEAR`；不会为了分叉删掉硬校验。

**两路 caption 可以不同。** 这是当前协议下合法的两条路线比较。若坚持后续 caption 完全相同，就必须显式修改协议／支持集，已经是另一项实验。

## 4.2 完整序列概率与类内选择

对每个规范 header 序列 $u_w$：

$$
\ell_0(w\mid h_0)
=\sum_{j=1}^{|u_w|}
\log\pi_0(u_{w,j}\mid h_0,u_{w,<j}),
$$

$$
q_0(w\mid h_0)
=\frac{\exp\ell_0(w\mid h_0)}
{\sum_v\exp\ell_0(v\mid h_0)}.
$$

各选项中的 WORD 按冻结的类内分布选择：

$$
\nu_{0,a}(w\mid h_0)
=\frac{q_0(w\mid h_0)}
{\sum_{v\in\mathcal W_a}q_0(v\mid h_0)}.
$$

这样不会因为研究者固定只用 `commit` 或只用 `locate`，额外重写选项内部行为。

必须使用序列 log-prob **求和**，不能按 token 数取平均。tokenizer 的空白、下划线、词边界和换行都属于契约；不能看到了下一 WORD 才反向寻找“共同前缀”。

但这个 $q_0$ 只是五条规范路径之间的分数。协议允许其他空白／大小写形式，模型还可能产生无效输出，因此：

$$
\boxed{
V_{\mathrm{nat}}
\ne
p_0Q_S+(1-p_0)Q_C
\quad\text{不能默认成立。}
}
$$

规范分叉与自然续跑应当分别测量。

## 4.3 最低 CPU／接口检查

未来实现只需围绕这一个契约检查：

- 两臂 prefix IDs、全部图像与 grid 完全相同；
- 已有 `crops=1`、实际 `observation_1` 和可用视图完整保留；
- 全局 `expected_step=2`，不沿用 fork 局部索引从 1 开始的假设；
- 强制 header 占用当轮与全局预算；
- 分别检查五条完整候选的 decode、parser、WORD/action 绑定；
- C 执行一次合法新增 crop 后恢复原自然策略，不限制其后续 recovery；
- 严格评分规则与原自然参照一致。

现有 first-action 代码中的 fresh `CropEnv`、单张原图和 `images[:1]` 都不能原样迁移。尤其辅助前向若仍只传原图，会丢失首裁回执，却表面上仍能计算出一个 loss。[decision_loss.py，第 193 行附近](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/5399cba8f4f8bb8d365b457a412409774dddaed2/pipeline/grpo/v52_dl/decision_loss.py#L185-L198)

这些检查证明接口对应所写的测量，不证明研究机制有效。

---

# 五、保留成功门控奖励，并正确处理历史成本

令已经执行 $n$ 次 crop，未来新增次数为 $M_a$，最终合法答对为 $Y_a$。继续保留：

$$
R_a=Y_a[1-\lambda(n+M_a)],\qquad \lambda=.01.
$$

于是：

$$
Q_a=(1-\lambda n)p_a-\lambda\mathbb E[Y_aM_a],
$$

其中 $p_a=P(Y_a=1\mid h,a)$。

对停止选项 $M_S=0$：

$$
\boxed{
D=Q_S-Q_C
=(1-\lambda n)(p_S-p_C)
+\lambda\mathbb E[Y_CM_C].
}
$$

在本卡的首裁后状态，$n=1$：

$$
D=.99(p_S-p_C)+.01\mathbb E[Y_CM_C].
$$

## 5.1 为什么历史成本不能直接删掉？

如果改成只算剩余成本：

$$
R'_a=Y_a(1-\lambda M_a),
$$

则：

$$
D'-D=\lambda n(p_S-p_C).
$$

历史 crop 的惩罚乘着最终随机 $Y$，并非动作无关的加法常数。

一个边界例子：$n=1$，$p_S=.49497$、$p_C=.5$，C 成功时新增一次 crop：

$$
D=.99(-.00503)+.005=.0000203>0,
$$

而删掉历史成本得到：

$$
D'=-.00503+.005=-.00003<0.
$$

效应很小，但排序确实可以翻转。

如果目标是 $Y-\lambda N$，历史成本才可以作为常数消去；那是另一个会处罚失败轨迹成本的目标。本卡不偷偷改成它。

## 5.2 四类情况应当用概率和完整后续价值解释

下面是**手算示例，不是实测结果**。均在首裁之后；表中 C 成功时的未来 crop 数固定。

| 情形 | $p_S$ | $p_C$ | C 成功时新增 crop | $D$ | 合理方向 |
|---|---:|---:|---:|---:|---|
| 继续有明显收益 | .20 | .80 | 1 | −.586 | 保留继续 |
| 成功率相同、后续调用无增益 | .80 | .80 | 1 | +.008 | 轻微倾向停止 |
| 继续路线具有误导性 | .80 | .30 | 1 | +.498 | 倾向停止 |
| 多步探索多数仍失败，但提高成功机会 | .10 | .35 | 3 | −.237 | 保留完整探索链 |

最后一行尤其重要：C 仍有 65% 的失败率，却有正的探索价值。不能因为某一次续跑失败就统一处罚它。

两个必须保留的反例：

- **一次同对不能证明冗余。** 若 $p_S=.3,p_C=.9$，独立单次采样仍有 .27 的概率“两边都对”，但继续路线显著更好。
- **一次同错不能证明探索没用。** 上表最后一行，两路单次都错的概率可达 $.9\times.65=.585$，但继续的期望价值仍更高。

如果真实 $p_S=p_C=0$，现有奖励确实给不出停止偏好。新增动作项为零只表示没有局部收益监督，不等于原 RLOO 不会处罚失败轨迹，也不等于我们已识别出未来训练后可能出现的探索价值。

## 5.3 奖励之外必须分别报告的量

$$
\mathbb E[R]
=P(Y=1)-.01\,\mathbb E[YN]
$$

扣的是**成功路径成本**。真实资源消耗则包括失败路径。

因此应分别列出：

- clean 与错框条件下的准确率；
- 首裁之后的实际调用数 $\mathbb E[M]$；
- 生成 token、输入／图像 token、延迟与 GPU 时间；
- invalid、协议预算耗尽和未完成率；
- 错框后的最终成功率及必要多步恢复是否被截断。

奖励上升不能替代这些联合证据。

---

# 六、只有测量支持后，才考虑写回 base 的算法

当前不建议启动这一阶段。下面给的是一条条件化、可实现的形式，供导师判断研究方向。

## 6.1 主算法：用原始收益差，对原模型的选项分布作有限倾斜

不增加小头。继续使用原 LM 对五条完整 WORD 路径的 $q_\theta(w\mid h)$，定义：

$$
p_\theta(h)=\sum_{w\in\mathcal W_S}q_\theta(w\mid h).
$$

参考分布 $q_0$ 和分叉差值 $\widehat D$ 都 stop-gradient。局部目标为：

$$
\boxed{
L_{\mathrm{dec}}
=
\frac1B\sum_{i=1}^{B}e_i
\left[
-\widehat D_i\,p_\theta(h_i)
+
\tau\,\mathrm{KL}\!\left(q_\theta(\cdot\mid h_i)\|q_0(\cdot\mid h_i)\right)
\right].
}
$$

这里：

- $B$ 是预定题／轨迹槽位数；
- $e_i=0$ 的无可用后裁状态贡献零；
- 不按 $\sum|\widehat D|$ 归一化；
- 不按 WORD token 数取平均；
- 不把低权重成本信号重新放大到单位尺度。

若以后仍采用导师的 RLOO 主干，则：

$$
L_{\mathrm{total}}=L_{\mathrm{RLOO}}+\beta L_{\mathrm{dec}}.
$$

**任务奖励保持原式，但总训练 surrogate 明确改变。** 它不是 baseline 替换，也不是原 RLOO 的无偏重写。

## 6.2 它为什么是有界的选项改进，而不是无界线性 logit loss？

对一个可以独立优化的状态，解析目标为：

$$
q^*(w\mid h)
=
\frac{
q_0(w\mid h)
\exp\!\left[\mathbf1[w\in\mathcal W_S]\widehat D/\tau\right]
}{
(1-p_0)+p_0\exp(\widehat D/\tau)
}.
$$

因此：

$$
p^*=\sigma\!\left(\operatorname{logit}p_0+\widehat D/\tau\right).
$$

并且：

$$
L_{\mathrm{dec},h}
=
\tau\,\mathrm{KL}(q_\theta\|q^*)+\text{与 }\theta\text{ 无关的常数}.
$$

第一项 $-\widehat Dp_\theta$ 对 $p_\theta\in[0,1]$ 有界。它没有要求把 logit 推向无穷。

固定类内分布、将 $z=\operatorname{logit}p$ 视为自由标量时：

$$
\frac{\partial L}{\partial z}
=
p(1-p)\{-\widehat D+\tau(z-z_0)\}.
$$

这仍然可能在极端概率处有小梯度。**我没有把它包装成“解决饱和”。** 如果参考模型在后裁状态也极端确信地选错，有限幅度更新可能难以改变它；这应当成为是否值得内化的测量条件。

## 6.3 为什么约束五种 WORD，而不只约束 answer/crop 两组？

因为 $Q_S,Q_C$ 是按参考类内混合测得的。

如果停止组从“主要 commit”变成“主要 abstain”，两组概率可以不变，实际准确率却会改变。五类 KL 有：

$$
\begin{aligned}
\mathrm{KL}(q_\theta\|q_0)
={}&\mathrm{KL}(\operatorname{Ber}(p_\theta)\|
\operatorname{Ber}(p_0))\\
&+p_\theta\mathrm{KL}(\nu_{\theta,S}\|\nu_{0,S})\\
&+(1-p_\theta)\mathrm{KL}(\nu_{\theta,C}\|\nu_{0,C}).
\end{aligned}
$$

解析目标 $q^*$ 保留参考类内比例，只倾斜两组质量。这没有新增五个独立价值估计器。

但共享 LM 的有限更新不保证逐状态达到这个解；它只是一个清楚的拟合目标。

## 6.4 有限 K、mask 与参数共享的限制

使用原始 $\widehat D$ 的好处是：对于固定 $h,\theta$，且分叉估计满足 $\mathbb E[\widehat D\mid h]=D$ 时，

$$
\mathbb E[-\widehat D\nabla_\theta p_\theta\mid h]
=-D\nabla_\theta p_\theta.
$$

它避免了前面 BCE 的 $\mathbb E|\widehat D|$ 项。**这不消除方差，也不表示复用同一批标签多次更新后，整个训练过程仍有某种无偏保证。**

新增损失直接计分于完整 WORD 路径及必要分隔符。共同 header 的共同概率因子在归一化中抵消；caption、bbox、答案正文不接收这项直接损失。原 RLOO 的 token 支持按原配方保留。

但：

- prefix 表征仍参与反向传播；
- 共享 decoder／LM head 仍可能改变 caption 和作答；
- 冻结视觉编码器或缩小可训练参数范围只能减少干扰，不能证明功能隔离。

还有一个容易遗漏的盲区：五条规范路径的概率同时下降，归一化后的 $q_\theta$ 可以不变。因而五类 KL 看不到规范路径总质量下降或自然 invalid 增长。

未来若真的更新，需要约束／检查从 $h_0$ 开始的**自然 token 分布**以及保留的回答、recovery 续写。只写一个 KL 系数不够：在参考参数处 KL 梯度为零，第一次普通 Adam 步不会因此自动受到保护。实际信赖约束需要候选更新后的检查、缩步或拒绝。

最终验收必须让更新后的 **base 完全自然生成**；不能用外部阈值或强制 WORD 替它执行，再宣布自主行为已经学会。

## 6.5 为什么这轮不转向势能 shaping？

完整、无折扣 episode 中：

$$
\sum_t[\Phi(h_{t+1})-\Phi(h_t)]
=\Phi(h_T)-\Phi(h_0).
$$

只有全部终止类型——包括 invalid、长度和预算终止——具有适当的终端势能，例如统一为零，且整条 return 按相同规则处理，才能得到动作无关的边界项。

只在 crop 轮添加差值、遗漏终止项、或再按特殊 token mask 分发，不自动享有这一结论。当前还没有可靠的后裁价值测量，先加势能函数不会补上这个缺口。

---

# 七、最强简单对照、完整轨迹收益与停止条件

## 7.1 唯一必要的简单对照：原 WORD 分数的校准阈值

后续若要证明新学习有价值，最强的廉价对照应利用原模型已经存在的状态排序：

$$
z_0(h)=
\log\sum_{w\in\mathcal W_S}e^{\ell_0(w\mid h)}
-
\log\sum_{w\in\mathcal W_C}e^{\ell_0(w\mid h)}.
$$

控制规则是：

$$
g_b(h)=\mathbf1[z_0(h)+b>0].
$$

触发时强制停止，否则原样自然续跑。阈值只在校准题上选，包含“不干预”和“全部停止”两个端点，不使用稀疏手选网格削弱对照。

若使用上述 64 组 Probe，可预先按题组分成 16 组校准、48 组评价，clean／错框分层锁定。这个切分很小，只适合开发筛查；不能把评价标签拿回去调阈值。Stage 0 的旧 $z$ 是首轮分数，不能代替新的后裁 WORD 分数；新打分前向也要记成本。

本轮不另训练一个复杂候选。若连这个简单对照已经足够，研究贡献就应收缩到停止机会的测量与自主内化，而不是增加一个价值预测模块。

## 7.2 单点干预有精确收益式，多点优势不能直接相加

令 $T=Q_S-V_{\mathrm{nat}}$。对于一次性停止覆盖策略：

$$
\boxed{
J(g)-J(\pi_0)
=
\mathbb E[e(h)g(h)T(h)].
}
$$

这个等式成立，是因为进入 $h$ 之前策略相同，干预只发生一次，之后的续跑定义固定。

同一批 $S,C,\mathrm{nat}$ 结果还能分别检查：

$$
\mathbb E[gD]
\quad\text{与}\quad
\mathbb E[gT].
$$

前者检验选中的停止路线是否优于继续取证；后者检验它相对原策略是否有实际空间。两者不能互相替代。

部署到多轮，或者更新整个 base 后：

- 到达哪些状态会变化；
- caption、框与续跑策略会变化；
- 原参考下的 $Q$ 会变化。

因此不能把旧轨迹上多个局部 $D$ 相加，声称得到新整段策略的收益。

## 7.3 停止条件应预先写清

| 观测 | 决定 |
|---|---|
| 完整状态、图像、协议或预算不能对齐 | 技术性停止；不进入分叉 |
| 机械删除全部后续调用的成本上限也太小 | 资源性停止 |
| 停止造成明确准确率／recovery 伤害 | 否定该停止规则 |
| $D$ 有差异，但相对自然策略的 $T$ 没有有用空间 | 不进入学习；原策略可能已利用该机会 |
| 相对自然收益出现，但不伴随相应 $D$ 优势 | 暂停“停止机制”解释，检查格式和续写分布变化；不直接宣布原因已明 |
| 只有总体相关高，停止边界仍无法判断 | 不生成动作训练标签 |
| 简单阈值已解释可见收益 | 不支持额外状态模型或新机制主张 |
| 区间太宽、预算耗尽 | 记为未决并停止，不补 K／改阈值补过关 |
| 未来规范标签指标改善，base 自然行为却掉分或不省成本 | 内化失败 |

对于“保准确率”，应使用预先声明的非劣要求，例如：

$$
\mathrm{LCB}_{95\%}(\Delta\mathrm{Acc})\ge-\epsilon_A.
$$

**我没有把负 2 个点默认为用户接受的容忍度。** 若“保准确率”按严格不降解释，就令 $\epsilon_A=0$。区间包含零、不显著下降，都不能替代这个要求。

E1 已经反复用于开发，疑似重叠的 40 题及其图像重复也必须处理。若只有这些旧数据，本轮最多是开发诊断；不能升级成确认结果。

---

# 八、待证研究命题、最近邻原文与实际阅读范围

## 8.1 这张卡值得研究的命题

可以把研究命题写成三段：

1. **机会命题：**首裁真实回执后，存在原策略尚未利用的停止机会，能节省实际成本而不牺牲成功与恢复。
2. **可识别命题：**这些机会能够从当时可见的信息预测，且价值超过简单阈值或固定策略。
3. **内化命题：**局部动作监督能够改变 base 自主行为，并保住选框、作答与 recovery。

第一项若不成立，应停止这张停止卡。若实际图像审查表明证据已经存在，但合法停止续写仍频繁读错、映射错，才有理由另立“证据利用”问题。不能只凭两路都错就判定证据利用是原因；也可能是第一张 crop 本身没有取得证据。

## 8.2 原始近邻：哪些已经有，哪些只是迁移

| 工作 | 我核对到的原始机制 | 对本卡的约束 |
|---|---|---|
| **Stop-RAG** | 将迭代检索建成 STOP／CONT 决策，从完整轨迹的前缀多次作答构造价值训练；其 continuation target 含未来前缀的最大值 | “前缀估值后学习停止”已有直接近邻；不能把它的未来最大值标签当成当前固定策略 $Q^{\pi_0}$ |
| **CAM-DF** | 对预执行的工具排名前缀，用停止与最佳后续前缀的 payoff gap，加权训练 logistic 停止器 | 收益差符号＋幅度加权 BCE 已有明确先例；它在任何工具输出出现前决策，且使用加法成本，与本卡状态／奖励不同 |
| **AdaTooler-V** | 用参考模型每题有／无工具的多次结果构造 benefit score，再调节整条轨迹的工具奖励 | 支持 K11 的收益条件化思想；题级收益不能直接充当后裁逐状态收益 |
| **VisionThink** | 用低／高分辨率成功计数比例控制两类动作惩罚，多轮 GRPO 训练模型输出 | 是 reward 设计近邻，不是本卡的局部 WORD 价值监督 |
| **MM-PRM** | 多模态前缀续写估计 MC 成功率，并使用软标签训练过程模型 | 支持 K19；原文也指出 MC 分数混合了难度和策略不确定性 |
| **VideoSEAL** | 冻结 inspector 掌握终止与最终答案，训练 planner 获取证据 | 支持 K57 的职责拆分；不能直接证明自主 base 已学会停止 |

上述机制分别可在原文方法部分核对。[arxiv.org](https://arxiv.org/html/2510.14337v1?utm_source=chatgpt.com)

**目前没有成立的算法新颖性结论。** 停止决策、MC 估值、收益加权监督、KL 约束都是已有方法组成部分。本项目可能有价值的部分，是在真实多模态工具协议中，证明并测清后裁停止机会，以及它能否被原模型自主实现。公式换一种写法不能替代这三项证据。

## 8.3 确实读过的材料

| 类别 | 本轮实际读取范围 |
|---|---|
| 新任务包 | `README.md`、`LESSONS.md` 全文；`audit_stage0.py`、`AUDIT.json` 全文 |
| Stage 0 | 固定 `5399cba…` 的 README、`stage0.py`、`stage0.json` 全文 |
| 历史解释与结果 | `MENTOR_INTERPRETATION.md`、三轮失败总报告、上一轮 `READOUT.md` 全文 |
| 上一版设计 | `PRO_RESPONSE.md` 中状态定义、冻结续跑、混合控制、阈值对照及 Stage 0／MVE／停止规则相关章节；未将它当成本轮效果证据 |
| 机制卡 | K19、K11、K57 全文 |
| Patterns | 父／子总表；`diagnose_then_target`、`measure_example_utility`、C07、C11 全文 |
| 代码 | v5.1/v5.2i 协议、v3 parser／hard checks、crop env／contracts；`easyr1_caption.py` 的分叉与真实 rollout 状态推进；`dl_audit.py` 导出段；`decision_loss.py` 辅助输入与更新接线段；`eval_v4.py` 记录／续跑段 |
| 论文原文 | Stop-RAG 的 MDP、训练与标签构造；CAM-DF 的问题定义、理论目标和 Eq. 6；AdaTooler-V §3.3；VisionThink 多轮训练与奖励；MM-PRM §3.3–3.4；VideoSEAL §3.4、§4.1–4.4 的相关部分 |

## 8.4 未取得或未完成核验的材料

- 私有原始 `audit_states.json`、`score_ref.json`、训练 fork JSONL；因此没有从原始记录重跑全部 bootstrap、验证采样独立性或重新验收所有唯一性检查。
- 真实模型权重、tokenizer／processor 的运行对象及服务器实际 import hash；没有执行模型前向或实际生产 loss。
- 后裁完整 token／图像 ledger；没有亲自逐图审阅该新状态样本。
- 其余 55 张机制卡全文及其全部原论文；没有声称本轮精读了 58 张卡。
- 其余父／子 patterns 及其全部底层论文、审稿材料。
- CounterCredit、TACO 等历史近邻本轮未重新核全文；不据此宣称已经完成全面查新。
- 导师内部对话原文件、音视频；本轮读取的是公开整理和其中的摘录。

---

## 可直接交给执行者的最小工作包

现在只需要三个产物：

1. **一份固定 reference 的自然轨迹状态清单**：题／图像组、预定 rollout 槽位、是否到达合法首裁回执、实际 token／图像／环境能否恢复、未来成本。
2. **一份协议与预算核查结果**：五个完整 WORD 路径是否合法、Step／图像／cap 是否一致；失败就停止。
3. **只有前两项支持时，提交一次 128 条 suffix 的无训练 Probe 申请**：明确它只是廉价否证筛查；不附带训练放行，不增加小头。

### 给导师的简短解释

> 首轮 Stage 0 已经完成了它的作用：当前首轮新增选路空间不足，所以停止小头。现在要测的是第一张真实裁图回执之后，继续取证还有多少价值。我们先保留同一个模型和原奖励，在合法的下一轮输出前比较停止与完整继续路线，并检查原模型是否已经利用了这些机会。只有存在足够、可识别的空间，才考虑把收益差有限地写进原模型的 WORD 选择。后裁空间、边界信号和自主内化目前都没有通过，暂不训练是有根据的决定。