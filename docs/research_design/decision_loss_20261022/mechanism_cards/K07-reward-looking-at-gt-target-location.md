# Reward or supervise where the agent looks using the training-time location of the target

Card `K07-reward-looking-at-gt-target-location` · status: candidate_pattern · members: 5 signatures from 5 full-text papers

## Operation

Compute a reward or label from the alignment between the agent's chosen view/region/span (camera axis, crop proposals, predicted boxes, accessed video span) and the privileged ground-truth target location, added to or gating the task reward.

- `2406.12505#s3`: add a perception reward lambda2 * exp(-delta_cam^4) that rewards pointing the camera at the next target, alongside progress, gate-pass, command-smoothness and crash terms
- `2406.19972#s2`: compute a look-at head orientation toward the target object's point-cloud center, solve the neck action by inverse kinematics, and mix it with the teacher's neck action as a_vla = (1-w_ar)*a_tch + w_ar*a_ar as the student's label (other joints follow the teacher)
- `2509.20912#s2`: Score each rollout with a composite reward: answer reward that rewards correct answers on positive/random instances, rewards 'Unknown' on counterfactual instances, penalizes guesses there and penalizes even more a correct answer without evidence; plus format reward; plus a region-coherence reward rewarding overlap of predicted boxes with evidence regions and penalizing overlap with irrelevant ones (zero on counterfactual instances)
- `2605.12571#s2`: multiply the 0/1 answer reward by a soft gate min(1, max tIoU(accessed span, gold interval)/gamma), gamma set to the training-set mean max-tIoU (about 0.05)
- `2505.21457#s1`: Score each sampled proposal set with a weighted sum of heuristic rewards (valid JSON/think format, pairwise-IoU non-overlap, box area 1-50% of image, fraction of GT boxes/mask covered) plus a task-aware reward obtained by running the fixed task model on each resized crop and computing AP/AR (detection) or SAM-mask mIoU (interactive segmentation); optimize with GRPO (group size 8, KL to base)

**Transfer notes:** Cross-domain join (active perception): robotics camera/head pointing rewards and VLM crop/box/span rewards share the operation and the information (GT target location, training only). Transfer needs region or interval annotations and evidence that is box-localizable; with tiny overlap scales the gate becomes a near-binary visit check.

## When it works

- Camera-pointing reward from simulator geometry lowers gate-passing error (0.516 -> 0.199 m on Ellipse) [2406.12505#s3]
- Look-at head label from the privileged object position mixed into the student's neck action: 67.9% -> 74.8% [2406.19972#s2]
- Region-coherence reward against detector-derived evidence boxes: +12.6 VSR, evidence mAP 35.5 vs 20.9 [2509.20912#s2]
- Soft tIoU gate on the answer reward against GT evidence intervals: LVBench 48.2 -> 50.2 [2605.12571#s2]
- GT-box coverage plus a task-model reward on the crops beats either reward alone for crop proposals [2505.21457#s1]

## When it fails

- Shown only on a state-based baseline and costs lap time [2406.12505#s3]
- Only +3.5 points on unseen tasks [2406.19972#s2]
- Gate saturates at very small overlap; decoupling answer authority contributed more [2605.12571#s2]
- SciQA decreases by 2.3 [2509.20912#s2]
- Task-model reward is wrong under domain gap; heuristic-only reward harms MMBench; small-object AP stays very low [2505.21457#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2406.12505#s3` — Demonstrating Agile Flight from Pixels without State Estimation (https://arxiv.org/html/2406.12505v1; Sec III-A4 Eq 3-4; Table I '+Perc' rows; Sec IV-A)
- `2406.19972#s2` — HumanVLA: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid (https://arxiv.org/html/2406.19972v2; Sec 3.2 'Active Rendering' Eq 5; Appendix E.2.4; Table 3; Table 4; Fig 10)
- `2509.20912#s2` — DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning (https://arxiv.org/html/2509.20912v4; Sec 3.3 Eq 5-8; Table 4, Table 6; Fig 3)
- `2605.12571#s2` — VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority (https://arxiv.org/html/2605.12571v1; Eq 6-7; Table 3; Sec 4.4, 5.3)
- `2505.21457#s1` — ACTIVE-o3: Empowering MLLMs with Active Perception via Pure Reinforcement Learning (https://arxiv.org/html/2505.21457v2; Sec 4.1, 4.4; Appendix B, C; Table 15; Tables 1-2)
