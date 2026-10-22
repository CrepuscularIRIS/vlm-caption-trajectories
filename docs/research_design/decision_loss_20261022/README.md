# Pro 设计入口：从优势估计转向动作选择监督

2026-10-22。用户授权公开本交接包并发送给网页 ChatGPT Pro。当前状态：**Pro 已完成，完整回复已收集**。本包不表示训练获批或研究假设成立。

## 阅读顺序

1. [设计任务与九项交付](PRO_DESIGN_BRIEF.md)：先读最新导师动作损失和 CC 补充说明。
2. [导师公式、视角转换与已有模式匹配](MENTOR_INTERPRETATION.md)：区分 reward、advantage、token 路由、动作辅助目标。
3. [当前结果与图文病例](../../30_rl_credit_report_20261022.md)、[旧 idea card](../../idea_cards/tool_credit_assignment_20261022.md)、[旧查新记录](../../idea_cards/tool_credit_assignment_scoop_20261022.md)、[RL 代码](../../../pipeline/grpo)。旧材料供比较，不替代最新任务。
4. [58 张机制卡](mechanism_cards/)：优先 K11、K19、K36；K17/K18/K22/K25/K46 作区分参考。完整卡已提供，按相关性阅读，不要求拼接所有卡。
5. [14 个父 pattern 总表](patterns/parent/overview.md)与[19 个子 pattern 总表](patterns/sub/overview.md)：两层全文均在相应目录。优先 diagnose_then_target、repair_degenerate_signal、manufacture_contrastive_pairs、reward_explicit_intermediates、decompose_and_reward_parts。
6. [来源与 SHA256 清单](SOURCE_MANIFEST.json)：保留抽取记录来源与本次发布版本。

[实际发送的提示词](SENT_PROMPT.md)与[Pro 完整回复](PRO_RESPONSE.md)已保存。回复是待核验设计建议，未启动新实验。

## 核心任务

固定 SFT-A 和 RLOO，评估同一真实前缀 h 上 answer / crop-then-continue 的收益差，直接监督完整动作标签的相对概率。推导原式、比较不同信用用途、设计最小 MVE。先解释“教模型选动作”与“重新计算优势”的区别，再给可实现的契约。一个主方案，最多一个必要区分对照。

## 证据边界

- 本包是历史机制/模式抽取与设计输入，不是完整论文原文、新的查新结论或训练结果。卡内历史数值、引文及评审归纳须回来源核验；录用类别不作为算法有效性的证明。
- 内部 Conversation、会议记录和私有运行文件未整库发布。解释稿保留与本问题相关的导师摘录；未公开内部路径只能当来源标识，不能声称网页已读取。
- CC 补充报告的 fc/zero 运行状态未在本次独立验证；它是替换裁图轮优势的方案，与旧手写两格规则、最新动作 BCE 分开讨论。
- Pro 输出属于待验证建模建议；不自动改实现、启动 GPU 或修改 Open Scientist 冻结状态。
