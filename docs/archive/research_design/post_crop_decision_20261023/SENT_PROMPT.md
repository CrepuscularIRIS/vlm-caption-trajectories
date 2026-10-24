继续上一轮设计，但先接受 Stage 0 的停止结果，不进入首轮小头 Stage 1。请通过公开 GitHub 链接读取本轮材料：
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/63c28ef843f1fa19f296eef0cf20a64cc8ebfa61/docs/research_design/post_crop_decision_20261023/README.md
其中链接了 Codex 工程教训与 CPU 复核、Stage 0、58 张机制卡及 parent/sub patterns。Stage 0 原始入口：
https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/5399cba/logs/dl/stage0/README.md
用户明确要减少第一次之后的冗余裁图，同时保准确率和 recover，最终是 base model 自主行为。本轮先形成可与导师讨论的一张主 card，再给条件化的最小无训练 Probe 与后续算法，不自动采样/训练。
请严格按新 README 的八项问题回答，尤其：不要把均值裁图>1当作冗余证据；后续 WORD 仍绑定动作，必须选择合法共同前缀；区分历史成本与成功门控奖励；不要把全局 D 可靠解释成停止边界可靠。核对勘误：5 个 D>0 中参考高度直答的是 2 个，第三个高 z 状态 D=0。
请实际读取和匹配已有 cards/patterns，不只从记忆套方法；原文未核实就标明，不宣称创新已成立。先总结导师建模思维与我们之前选错对象/测量的教训，再收敛成一张主卡，明确观察、假设、最小否证、梯度接收者、最强简单对照和停止条件。允许建议暂时不训练。请列出确实读过与未能读取的材料。中文回答。
