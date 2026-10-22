# 实际发送给网页 Pro 的提示词

模式：Pro。发送日期：2026-10-22。材料固定于提交 `65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc`。状态：已发送，等待完整回复。

---

请用 Pro 继续算法建模，中文回答。导师最新转向：同一 caption 后、动作标签前的 h，分别 answer 与 crop-then-continue 多次续跑，用收益差加权 BCE 直接监督完整动作标签，并加到 RLOO。

请亲自读取新 GitHub 交接入口，按其中 PRO_DESIGN_BRIEF 九项交付：
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/tree/65b7d24b5f9336d1ec76ed7f10a2a87f9a9025dc/docs/research_design/decision_loss_20261022

入口已链接：导师公式与解释、旧 idea card、当前图文结果、训练代码、58 张机制卡全文、14 父/19 子 pattern 全文。请先读 README、PRO_DESIGN_BRIEF、MENTOR_INTERPRETATION，优先 K11/K19/K36 和相关 pattern，再回原文核关键论断。页面打不开可用同路径 raw；列实际已读/未读，不用旧对话代替新资料。

重点独立比较 advantage/baseline、reward shaping、token 信用路由、动作 BCE 的目标与梯度；区分正在运行的 fc/zero 与新方案。给一个主算法、最多一个必要区分对照，含 K=4 噪声、完整标签概率、mask/reduction、精确推导、伪代码、CPU 检查及最小 MVE/反证/预算。先用人话解释导师的视角转换。卡匹配不是新颖性证明；不强行拼卡，不以保持期望梯度为硬约束。只设计，不执行训练。
