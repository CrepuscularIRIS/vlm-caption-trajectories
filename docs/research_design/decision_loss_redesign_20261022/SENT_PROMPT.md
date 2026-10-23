请用 Pro 重新设计上一轮 VLM 裁图决策方案。三个实验已经完成，都未达到保准确率少裁图的目标。请先读取新的公开任务包，按其中要求独立审查失败机制、复用 idea cards/patterns，给一个主方案和一个最小匹配对照，不执行训练。

https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/5c63a2803f3a4fe2f5105e461aafc780aa03e031/docs/research_design/decision_loss_redesign_20261022/README.md

里面链接了 ba449ba 的真实实现、PREREG、逐状态结果，以及完整卡库和你上一版设计。必要时使用同路径 raw 或 GitHub 文件接口。请特别检查：正确且自信的裁图状态上 BCE 梯度小是否真是缺陷；全局直答偏移、共享参数干扰、有限 K 噪声怎样区分。不要直接接受报告“原因已查明”的措辞，也不要只把 BCE 改成无界线性 loss。请真正读选中的卡和关键原文，说明哪些只是类比。中文回答，先人话解释，再推导和最低成本可证伪 Probe，最后列实际已读/未读材料。
