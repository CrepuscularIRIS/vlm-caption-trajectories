# Pro 新会话交接：减少中间多余步骤，保持准确率与 recover

本包对应用户的新指示，取代上一轮“是否先恢复首轮直答”的提问框架。**目标已定：本可两步完成的题，不应走四五步；直答少不是本阶段必须修复的问题。** 不以恢复首轮直答率、重新制作 SFT 或全面解决感知错误作为前置目标。卡库是方法参考，不是研究路线的上级指令。

本包只做文档交接与网页 Pro 咨询。不启动、停止或调整训练，不改变正在执行的验收规则。已有 λ=.05/.10 两组仅按 docs/33 的“已启动、待结果”记录提供给 Pro；本轮没有验证服务器进程，不能把该记录写成已完成实验。

## 阅读顺序和证据权限

实验库固定提交 [9a1af8d](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/tree/9a1af8d127b79b54bb16b57c2cd86a719f2784a0)。后面的链接均是该快照。原始 lark 已在本地回读；公开交接仅用已经公开的指导记录和实验文档。原 Conversation 没有逐消息时间戳，日期沿用实验/报告标记，不能称核实了消息发送时间。

1. **最新主入口**：[docs/33](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/33_status_20261023.md) 全文。
2. **导师要求与基础路线**：[docs/20](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/20_mentor_guidance_20261010.md)、[docs/23](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/23_mentor_guidance_20261016.md)、[docs/24](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/24_stage_report_20261018.md)。旧计划服从用户最新目标及 docs/33；尤其不能把旧文档“恢复直答”重新设为 KPI。
3. **10-20 后核心问题与实验**：[docs/29 两分支与 dV](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/29_twobranch_dv_grpo_20261020.md)、[docs/30 RLOO 与信用实验](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/30_rl_credit_report_20261022.md)、[docs/32 历史及旧 Pro 回合](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/32_research_trajectory_20261023.md)、[Stage 0](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/logs/dl/stage0/README.md)。docs/32 是历史背景，非当前框架；不要再复活已被停止规则挡下的首动作预测头。
4. **较强效率信号的旧参照**：[docs/27](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/27_results_20261019.md)，并看 docs/23 §9。注意卡 B 的 −37% 与 −25% 使用了不同原版参照，不能混写。
5. **新零 GPU 测量**：[logs/stageA](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/tree/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/logs/stageA)、[超额裁图汇总](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/logs/later_crops/excess_160.json)、[计算脚本](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/pipeline/eval/later_crops/excess_crops.py)。读结论时同时看 `ledger_diag.py`、`push_later.py`、`test_pbrs_rloo.py` 的真实计算对象。
6. **直接看图审查作为局部证据**：[10条/49轮记录](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/research_design/trajectory_root_audit_20261023/direct_review/README.md)；同目录 TIMELINE.md 和上一级 CASES.md 包含全部文字/图片链接。没有读取图片就只称参考 Codex 注释。分类只帮助选测量位置，不升级为覆盖所有失败的总框架。
7. **新版 corpus**：[领域理解](https://github.com/CrepuscularIRIS/vlm-caption-corpus/blob/21b7650c15693b4af5ca94d314e0104983b7bd10/library/UNDERSTANDING.md)、[92卡完整导航及5工程卡](CARD_INDEX.md)、[92卡操作摘要](https://github.com/CrepuscularIRIS/vlm-caption-corpus/blob/21b7650c15693b4af5ca94d314e0104983b7bd10/library/mechanism_cards.jsonl)。先浏览完整目录，再精读相关卡；不声称一次咨询核完了499篇原文。私有 ROADMAP 不在本次网页输入中，也不复述其私有实验细节。

## 从导师要求到已做实验：需要整理的主链条

| 问题/建议 | 已做与未识别的部分 | 本轮 Pro 应回答 |
|---|---|---|
| 主动观测过头，保持准确率与 recover，缩短轨迹 | 强代价组减少裁图，但单种子、区间含0不等于非劣 | λ剂量组现在能检验什么，哪些情况需要后续信用方法？ |
| 固定A SFT，同题两个分支各4次，检查观测需求 | dV有数据，但首轮统一加差值未改变所需行为；后裁反增 | 将首轮实验得到的信息与中间步骤问题区分，保留可复用证据 |
| 干净RLOO λ=0/.01，其余一致 | 不再出现原标准化下的塌缩，但效率未改善 | 优势估计器问题与效率目标力度问题分开 |
| 四种前后对错，决定哪些token接收结果 | 四格规则、冻结读者、fc、zero存在不同限制；不能统称已测试所有信用分配 | 只选择与中间额外步骤有直接关系的信用机制 |
| 同一合法prefix比较两动作价值，直接监督动作选择 | 首动作监督未取得目标效果，Stage0首轮空间小 | 若移到后续回合，合法分叉必须在WORD提前承诺前；当前并不自动授权重跑 |
| 后续考虑势能方程 | 轨迹级严格PBRS会抵消；逐轮不等于天然减步 | 势函数是降方差还是改变偏好？需要何种证据才能值得实现？ |
| 实验报告要求 | 分benchmark、token、更新、rollout/GPU开销；KL/DS改变须说明 | 建议一次简单、可解释的对照，不叠满多种机制 |

## 请 Pro 审查 docs/33 的解释，不改变目标

1. **样本最短参照的界限。** 四次自然采样中最短正确样本的裁图数，可定义经验额外长度；其余路径的上下文/错误定位不同，不能称它们都能从相同中间状态少走这些步。随机猜对也可能是最短“成功”。14%是该统计量占所有裁图的比例，不是已经验证的可删除比例；所谓下界必须说清相对于哪个定义、是否要求可靠正确。
2. **2–5%不是参数梯度份额。** 当前计算是 `sum(abs(A_cost)*gen_tokens)/sum(abs(A)*gen_tokens)`。它是优势幅度代理，不含score梯度、优化器、clip或共享参数方向；足以提出“代价偏弱”假设，不能独立证明唯一瓶颈。
3. **长度与正确性的相关不是删步的效果。** `sum A_i(N_i−mean N)` 是优势—长度关联；实际参数更新后期望长度的方向还涉及梯度内积、参数化、token聚合与优化器。困难路径失败后多试也会产生负相关。请明确推导可推到哪一步。
4. **多checkpoint有用，但不等于多seed。** 跨步波动可能包括真实学习动态与评测噪声；四个相关checkpoint并非四次独立复现。探索时可看趋势，不能把“3/4过线”直接当独立显著性。
5. **非显著与非劣不同。** 当前2百分点门是已登记的操作性选择规则，本轮不改它；请建议如何同时诚实报告区间、逐题对→错回归、E2和各benchmark差异。新增统计建议如会影响原判定须标为探索性，不能事后改门。
6. **全错轨迹和recover。** 原奖励 `Y(1−λN)` 对全错组不能按长度定优劣；但不能因失败就否定探索。E2只是首次偏移压力测试，还未覆盖导师说的全部不确定性。方法不能用失去recover换“少步”的表面胜利。

## 所需输出

- 先给完整读取清单：已读全文、仅索引/摘要、无法访问分开。
- 一页“导师核心疑问→实验回答→仍缺什么”，将 docs/33 作为主入口。
- 最多两条与中间步骤直接相关的思路，并推荐一条；既有 λ 对照若已是最有价值的下一步，可以建议等结果，不为新颖性硬加模块。
- 每条思路引用具体卡/成员前作，区分应用改造与潜在贡献；给状态、可获得信息、操作、信号、梯度token、是否改目标、最简单对照、失败条件。
- 将即将到来的 λ=.05/.10 结果分成三种情况，说明分别支持什么、下一步只做哪项最小检查；不能把它们当成已有结果。
- 默认后续仍为 probe/MVE，优先已有数据。设计不是训练授权，不追加GPU工作、不修改正在运行的规则。
