请继续上一轮研究设计。用户授权你依据已有轨迹的实际图像审查，重新设计下一阶段的一张主 idea card，先不训练。请实际读取这个已发布、固定commit的审查包：
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/0dfac43aed8f65a641f9a322c5f37863d8d84579/docs/research_design/trajectory_root_audit_20261023/README.md
逐图和逐轮证据：
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/0dfac43aed8f65a641f9a322c5f37863d8d84579/docs/research_design/trajectory_root_audit_20261023/CASES.md
已有 cards/patterns 入口：
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/0dfac43aed8f65a641f9a322c5f37863d8d84579/docs/research_design/decision_loss_20261022/README.md
请先按新README核对数据口径和三个反例，再逐项回答其中8个设计问题。所有本地材料已放公开链接，不需要本地连接。若GitHub页面提取失败，可读取相同commit/path的raw.githubusercontent.com版本；图片不能打开就明确没有亲自核图。
这次要回答“根本可能错在哪个环节”，而非自动沿用stop损失：可获取性、定位、证据读取/选项绑定、停止分别有什么证据，哪些不能从自然轨迹识别？上一轮首裁后可停止性仍是候选，允许保留、收窄或暂缓。我们要求最终base model自主、减少无效截图同时保持答题和recover；不把外部检查器当最终方案。
请实际匹配1–3张已有机制卡及patterns，不要只列名字；先给零新采样的最小区分，再给条件化无训练probe和可能的目标/梯度接收者。新采样/训练仅作为建议，不是执行授权。不要从10条非盲审案例估计全库比例，不要把E2都当破坏，不要把同题不同采样当相同状态反事实。不要把正确答案当论据正确。
最后用导师能读懂的中文总结：下一步先辨别什么，何种结果才值得训练，什么结果应停止。列出实际读到与未能读取的材料及未核实的创新性。