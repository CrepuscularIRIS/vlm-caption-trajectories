请在这个新 Pro 对话里重新整理我们 VLM 多轮工具 RL 的研究主线。用户已明确纠正目标：直答少可以是合理的；我们要减少的是中间额外步骤——本来2步能完成，现在却花4–5步，同时尽量保持准确率、不损害不确定时的 recover。请不要再把“恢复首轮直答支撑”作为主线，也不要把末轮读数纠错替代减少中间步数的问题。

请先完整阅读这份固定版本的交接包，它逐项链接导师公开指导、10-20之后的实验、当前docs/33、零GPU统计和代码、以及新版corpus：
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/a0c597f875cc79fdea29ea8c1db3b19cf12c8d1e/docs/research_design/later_crop_corpus_review_20261024/README.md
若GitHub渲染取不到正文，使用原始文本：
https://raw.githubusercontent.com/CrepuscularIRIS/vlm-caption-trajectories/a0c597f875cc79fdea29ea8c1db3b19cf12c8d1e/docs/research_design/later_crop_corpus_review_20261024/README.md

当前主入口docs/33务必全文阅读：
https://raw.githubusercontent.com/CrepuscularIRIS/vlm-caption-trajectories/9a1af8d127b79b54bb16b57c2cd86a719f2784a0/docs/33_status_20261023.md
还要沿交接包依次看docs/20、23、24的导师要求，docs/29、30的实验、docs/32的旧路线及Pro回合、logs/dl/stage0、logs/stageA、excess_crops.py。请理解导师为什么先查advantage、再问四种情况如何分信用、再试动作选择约束、再提势函数。不是要求把所有方向叠起来。原始lark已由Codex本地回读；网页输入只提供已公开的指导与实验记录，不猜不可访问的私有文件或消息时间。

新版corpus：
https://raw.githubusercontent.com/CrepuscularIRIS/vlm-caption-corpus/21b7650c15693b4af5ca94d314e0104983b7bd10/library/UNDERSTANDING.md
完整92张机制卡+5张工程卡导航：
https://raw.githubusercontent.com/CrepuscularIRIS/vlm-caption-trajectories/a0c597f875cc79fdea29ea8c1db3b19cf12c8d1e/docs/research_design/later_crop_corpus_review_20261024/CARD_INDEX.md
全部92卡的操作/信息/条件摘要：
https://raw.githubusercontent.com/CrepuscularIRIS/vlm-caption-corpus/21b7650c15693b4af5ca94d314e0104983b7bd10/library/mechanism_cards.jsonl
请先完整浏览卡的目录/摘要，再精读与你最终建议相关的机制全文与工程卡，并核查关键成员论文。卡库是参考，不是让我们服从“保直答→效率→信用”的ROADMAP。无需声称核完499篇原文；必须区分全文、摘要、未读与打不开。

这次请尤其避免延续旧推断：
- 同题4次自然采样中最短正确样本，只是经验长度参照，不是相同中间状态的反事实最短路径。14%额外裁图不是已经证明可无损删除的14%。
- λ项的2–5%是绝对advantage乘token的幅度代理，不是参数梯度份额；长度和正确性负相关也不能单独证明删步会改善。
- 旧强代价组的准确率差异不显著，不等于已证明非劣；四个checkpoint有真实学习动态，不能当四个独立seed。
- docs/33中的λ=.05/.10已经由其他执行者启动，但本轮没有结果，不要重复建议启动、不要修改它们事先的判读规则。可以补充标为探索性的分析建议。
- 逐图审查是用来选中间诊断位置的局部证据，不是另建一套覆盖所有视觉错误的研究框架。若图片没读到，不能声称亲自看过。

请完成阅读后再给完整中文整理：
1. 一页人话：导师核心问题→我们具体测过什么→哪些判断成立/仍只是推断。主目标始终围绕同等答题与recover下缩短中间轨迹。
2. 审查docs/33现在的结论、指标和判读：剂量对照能回答什么，回答不了什么。区分“代价力度”“给哪些token信用”“后续停/继续选择”和“裁图定位效率”。
3. 从92卡中筛出最多两条值得考虑的思路，推荐一条；允许建议先等现有λ对照，不为了交付idea硬造新方法。对每条明确匹配卡ID、最近前作、适用条件、真正差异；应用改造就直说应用改造。
4. 若推荐方法：给简洁算法，定义合法中间状态h、当时可见信息、可选操作、估值/奖励、哪些token接受哪项loss、是否改变目标、如何避免全局少裁偏置与recover退化。PBRS需写清终止/折扣/按轮回报与轨迹级RLOO的区别，不把纯telescoping当效率机制。
5. 对现有λ实验的三类结果分别给一个最小后续动作：少裁且精度可接受、少裁但掉分、仍不少裁。优先已有轨迹/日志，缺数据才提无训练probe；不要默认四臂训练。简单对照、失败条件、开销、开发与确认集隔离说清楚。非劣界限是研究者决策，不替用户升级现有2点操作性门为科学非劣结论。

最前面给实际读取清单；最后给可转给导师的短总结。请不要只输出检索计划或再让我们选择一大堆card。此次只咨询设计，不授权执行实验。