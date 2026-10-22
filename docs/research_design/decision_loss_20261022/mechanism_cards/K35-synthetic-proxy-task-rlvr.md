# Run RLVR on a self-supervised proxy task whose answer comes from the construction

Card `K35-synthetic-proxy-task-rlvr` · status: candidate_pattern · members: 3 signatures from 2 full-text papers

## Operation

Construct a proxy task whose answer is known from how it was built (corruption parameter, the span of an injected caption error) and optimize it with GRPO to train perception without downstream labels.

- `2510.16416#s1`: Apply a known corruption to each image, ask the VLM to predict the corruption parameter in natural language, reward exact correctness (0/1), and run GRPO (group 5, KL 0.01) on these rewards
- `2510.16416#s2`: Increase corruption difficulty (3x3 grids, 45-degree rotations, stronger augmentation p=0.8 and crop scale 0.3, DINOv2-retrieved hard negatives) to match the SSL task to model capacity
- `2506.10128#s1`: Use an LLM to minimally perturb one noun-phrase description in a long human caption into a visually similar plausible error, ask the VLM to output the hallucinated span, and reward exact (relaxed-boundary) string match plus format with GRPO

**Transfer notes:** n/a (in-domain).

## When it works

- Rotation/position pretext rewards: +7.4% MMBench at 3B [2510.16416#s1]
- Raising pretext difficulty helps larger models for some tasks [2510.16416#s2]
- Locating one injected error in a dense caption: CHAIRs and general average improve at 7B and 72B [2506.10128#s1]

## When it fails

- Diminishing at 7B; combining rewards gives no gain; train/test image overlap not stated [2510.16416#s1]
- Harder rotation and jigsaw are detrimental [2510.16416#s2]
- Spatial and text error detection drop; marginal at 72B; no comparison to other proxy data [2506.10128#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2510.16416#s1` — SSL4RL: Revisiting Self-supervised Learning as Intrinsic Reward for Visual-Language Reasoning (https://arxiv.org/html/2510.16416v4; Sec 3; Sec 4.1 Table 1; Sec 4.2 Table 2; Sec 4.3 Table 3; Appendix G Tables 22-23)
- `2510.16416#s2` — SSL4RL: Revisiting Self-supervised Learning as Intrinsic Reward for Visual-Language Reasoning (https://arxiv.org/html/2510.16416v4; Sec 4.5.4 Figure 4; Appendix C (referenced); Appendix J Tables 29-30)
- `2506.10128#s1` — ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs (https://arxiv.org/html/2506.10128v1; Sec 3; Table 1; Table 2; Sec 5.1-5.2)
