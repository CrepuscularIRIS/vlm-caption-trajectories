# Regress training-only targets from the model's own representation through discarded auxiliary heads

Card `K06-auxiliary-heads-on-training-only-targets` · status: candidate_pattern · members: 3 signatures from 2 full-text papers

## Operation

Attach auxiliary heads or learnable queries to the model's internal states and regress targets available only at training (privileged robot state; frozen vision-foundation-model features), dropping them at inference.

- `2504.10390#s2`: attach an auxiliary head sharing the first layer(s) of the student actor and train it with a regression loss to predict the auxiliary observations from proprioception, added with a weight to the student PPO + adversarial loss; discard the head at deployment
- `2510.14349#s1`: pass the MLLM's output MTQ states through a per-VFM Transformer alignment layer (VAQ cross-attends to MTQs) and add a discriminative MSE + InfoNCE loss to that VFM's features alongside the next-token loss, training the projector, MTQs, VALs and (in SFT) the LLM
- `2510.14349#s2`: mask attention so that queries supervised by different VFMs are mutually invisible during causal processing

**Transfer notes:** Cross-domain join: robotics auxiliary privileged-state heads and MLLM auxiliary VFM-feature alignment share operation and information (training-only targets, head discarded). Transfer to VLM RL needs per-image targets (depth, masks, boxes) and has not been tested under RL.

## When it works

- Auxiliary regression of privileged observations speeds student convergence [2504.10390#s2]
- Aligning MLLM task-query states to DINOv2/depth/segmentation features: MMVP 20.0 -> 37.9, HaluBench 47.5 -> 57.2 [2510.14349#s1]
- With several targets, mutually masked query groups are needed [2510.14349#s2]

## When it fails

- Robot evidence is learning-curve speed only [2504.10390#s2]
- POPE slightly lower; token-level alignment variants hurt MMBench/SEED; bounded by the encoder; no RL post-training tested [2510.14349#s1]
- Without the mask, multi-target training is worse than single-target [2510.14349#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2504.10390#s2` — Teacher Motion Priors: Enhancing Robot Locomotion over Challenging Terrain (https://arxiv.org/html/2504.10390v2; Sec II-C Eq 11; Sec III-B Terrain Level Convergence, Fig 4)
- `2510.14349#s1` — Vision-Centric Activation and Coordination for Multimodal Large Language Models (https://arxiv.org/html/2510.14349v3; Sec 3.1, 3.3; Table 1; Table 4(a); Tables 2-3)
- `2510.14349#s2` — Vision-Centric Activation and Coordination for Multimodal Large Language Models (https://arxiv.org/html/2510.14349v3; Sec 3.2; Table 4(b))
