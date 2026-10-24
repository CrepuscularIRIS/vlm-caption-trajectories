> 公开删节提示词；本地保留完整发送文本。

请在这个全新对话中，独立设计下一阶段 VLM 工具使用 RL 的研究思路。请用 Pro 做实质建模，中文回答。不要沿用旧对话结论，也不要只替我们概括文献。目标是在尽量不降准确率、不损害不确定时的 recovery 的前提下，减少无效裁图/轨迹长度；当前只做设计与最小诊断，不授权训练。

请实际打开并阅读以下 GitHub 资料，先报告读取成功/失败的文件。公开 corpus 固定版本 21b7650c15693b4af5ca94d314e0104983b7bd10：
https://github.com/CrepuscularIRIS/vlm-caption-corpus/blob/21b7650c15693b4af5ca94d314e0104983b7bd10/library/UNDERSTANDING.md
https://github.com/CrepuscularIRIS/vlm-caption-corpus/tree/21b7650c15693b4af5ca94d314e0104983b7bd10/library
仓库有 92 张机制卡、5 张工程卡和 499 份研读记录。卡有 candidate_pattern/case/needs_revision 之分；论文数量、卡片标签和整理质量不等于机制已在本项目成立。请从索引选择最相关的 3–5 张卡细读，追溯关键成员论文的原文位置；未核实论文要标清楚。优先 PC-F3-02、PC-F1-09、PC-F2-01、PC-F2-20、PC-F5-04，并自行挑选比它们更有解释力的卡。公共 cards 路径是 library/cards/F1_objective、F2_credit、F3_support、F4_perception、F5_stop_eval。可用同路径 raw.githubusercontent.com 读取，不能把一个链接打开失败当成全部资料不可读。

项目实证和逐图审查入口：
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/3d7a7ba60922ef75205c5af8539356608efc4389/docs/research_design/trajectory_root_audit_20261023/direct_review/README.md
同目录 TIMELINE.md、step_annotations.json、questions.json 保留49轮原文/完整选项/实际框/图片引用；上一级 README.md、statistics.json、extended_statistics.json 有抽样口径和旧报告更正。请读这一组，不能只引用“后来裁图约多少”而忽略动作和图像。看不到图就明确只读了审查文字，不能声称亲自看过。
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/main/logs/dl/stage0/README.md
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/main/docs/30_rl_credit_report_20261022.md

[公开版省略私有来源链接、路线摘要及读取请求；本段影响了 A 的输入背景，详见对照说明。]

需要你处理的核心分歧：
1. 首轮直答少，究竟是支持集/采样塌缩，还是当前题池与奖励下合理选择？stage0提示首动作路由空间很小，而后续裁图还有空间。必须区分“首轮不用图就答”与“取过证据后及时停止”；首轮保支撑是否真是所有后续信用方法的必要前提？零次观察到动作不等于数学概率为零，也不等于能力消失。自然采样与强制动作续写各识别什么？
2. 逐图案例中，错误来源包括：中途把目标行认错后继续裁；正确知道缺信息却选无法补信息的crop；已明确写出正确关系仍确认；最终字母正确但解释与选项矛盾。怎样用已有卡构造可区分的模型，而不是统称 credit assignment？末轮纠正数字不能解释此前裁图长度。
3. 请重新推导目标/advantage/信用路由/辅助损失/采样支撑之间的关系。检验成本放入GRPO标量奖励时究竟是被消掉、被重标度还是被正确性压过；mean-only在混合组是否也保留成本项。PBRS抵消需写出gamma、终止势、有限时域、共同前缀、截断和return-to-go条件。BCE的梯度应按标签方向推导，别笼统把所有失败叫饱和。兄弟留一是否仅降方差，还是在强制分叉/分段/裁剪后改变优化目标？强制分叉进入loss不自动等于无偏，需要明确行为概率、动作token与后续token的校正。
4. 有哪些方案看似新其实已被卡片成员论文覆盖？不要因组合两张卡就叫创新。用户还提示 AdaTurn、2610.12060、SAGE引用与评测集重叠有待核查，未见原文不要下定论。

请交付：
A. 用人话说明当前最值得追的一个问题，以及与“只调advantage”“只改信用落点”“只学停止”的区别。
B. 最多三个竞争解释，每个写出现有证据支持什么、不支持什么、一个可推翻它的观察。
C. 一张主idea card（最多一张备选）：匹配具体PC卡ID、最近前作、真正的差异/或坦承只是实现控制；明确定义状态、动作、可获得信息、训练信号、哪些token接收哪项loss、是否改变目标。不要强行造方法名。
D. 一个按决策价值排序的MVE：先仅用已有材料的可行检查，再列“缺数据才需要新推理”的最小条件测试；不能把pass@64、AA重复训练或模型重新打分叫作无需GPU，除非相应样本/分数已存在。只选最能区分解释的一项，别默认跑四个大臂。给简单基线、可观察在线分层、保护recover的指标、非劣界限如何由用户确定、成本和停止条件；评测题/同源图不能回流训练。
E. 明确建议现在先推进什么、暂停什么、哪项新证据会让你改变选择。请不要以“再调lambda/再加loss”代替建模，也允许结论是现有方法不值得继续。请完成整个分析再给最终答复。