# Give the reward model or judge training-only ground truth that the policy never sees

Card `K04-privileged-evaluator-reward` · status: candidate_pattern · members: 2 signatures from 2 full-text papers

## Operation

Condition the reward model or LLM judge on privileged ground-truth text (human captions/rationales, GT tool-call annotations) while the policy sees only image and prompt, and use its scores as RL reward.

- `2309.14525#s1`: concatenate human-written factual descriptions of the image into the reward model's input, train it on the same binary hallucination-aware preference pairs, and use it as the PPO reward
- `2509.14480#s1`: score each turn -1/0/1 with the judge (at most one -1), scale -1 by 5x and other turns by 1/T, add 10x the binary terminal reward, and use the total as a single trajectory-level reward for GRPO/PPO advantages

**Transfer notes:** Direct VLM-side instance of brief mapping 2 (asymmetric actor-critic). Holds when annotations exist at training time only and cannot leak into the policy input. Reward quality is bounded by annotation coverage.

## When it works

- Reward model reading 5 COCO captions reduces verbose hallucinated responses under PPO (MMHal 1.8 -> 2.1, MMBench 48.9 -> 51.4) [2309.14525#s1]
- Turn-level LLM judge with GT tool-call annotations, aggregated into one trajectory reward: +6% relative pass^1 over GRPO [2509.14480#s1]

## When it fails

- Requires GT captions/rationales for every RL image; POPE slightly drops; short or evasive responses can score high [2309.14525#s1]
- Placing per-turn rewards at each turn's final token under PPO/GAE degrades below vanilla PPO; out-of-domain generalization deteriorates [2509.14480#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2309.14525#s1` — Aligning Large Multimodal Models with Factually Augmented RLHF (https://arxiv.org/html/2309.14525v1; Sec 2.4; Table 5; Sec 3.6)
- `2509.14480#s1` — Process-Supervised Reinforcement Learning for Interactive Multimodal Tool-Use Agents (https://arxiv.org/html/2509.14480v1; Sec 3.2; Sec 4.2; Table 2 (text); Sec 5.1 Fig 4)
