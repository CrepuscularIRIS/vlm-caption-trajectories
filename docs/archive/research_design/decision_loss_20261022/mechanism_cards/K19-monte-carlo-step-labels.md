# Label reasoning steps by sampling continuations and checking final answers

Card `K19-monte-carlo-step-labels` · status: candidate_pattern · members: 6 signatures from 4 full-text papers

## Operation

Estimate each step prefix's value by sampling continuations to a verifiable final answer (binary search for the first error, MC success rate, dual-completer agreement) and train a PRM on it or use it directly as reward.

- `2501.04686#s1`: binary-search for the first erroneous step by running N rollouts from the middle prefix and testing whether any reach the correct answer (mc>0), labelling steps before it correct and after it incorrect
- `2505.13427#s1`: Run OmegaPRM-style MCTS: roll out completions from step prefixes, use binary search to locate the first step after which completions stop reaching the correct answer, and record per-prefix Monte Carlo success rates in a state-action tree
- `2505.13427#s2`: Train with cross-entropy against continuous MC scores (soft labels) instead of thresholded binary labels, at a small learning rate (4e-6); at test time score each candidate's steps, aggregate (Min/Average/Max/SumLogPr/SumLogOdds/MeanOdds) and pick the best of N
- `2506.09532#s1`: estimate each step's hard MC label separately with a weak and a strong completer and keep only steps where both labels agree, discarding the rest
- `2506.09532#s2`: initialize the PRM from an ORM trained on large outcome-labeled data and up-sample solutions containing negative step labels (rate 2) when fine-tuning on step labels
- `2606.11209#s1`: for each step prefix of a sampled response, sample M continuations from the current policy, score each by final-answer match, average to a per-step solvability c_i, average c_i over steps to R_proc, gate R_proc below tau=0.5 to a negative step-bonus penalty, and combine with format reward and bounded step bonus before group-normalizing advantages

**Transfer notes:** n/a (in-domain). Bears directly on brief obstacle 1: these labels do not observe whether a step read the image correctly.

## When it works

- Binary-search labels improve PRM Best-of-N [2501.04686#s1]
- MCTS rollouts give a PRM lifting Best-of-16 33.92 -> 42.80 on MM-K12 [2505.13427#s1]
- Soft MC targets beat thresholded labels (43.0 vs 34.4) [2505.13427#s2]
- Weak/strong completer agreement raises label accuracy to 94% at ~45x less compute [2506.09532#s1]
- ORM initialization and negative up-sampling add <1 point [2506.09532#s2]
- Per-step solvability as the only reward: 59.72 vs 57.55 outcome-only on video [2606.11209#s1]

## When it fails

- Labels encode answer reachability, not perceptual correctness; after-first-error labelling induces a length bias [2501.04686#s1, 2606.11209#s1]
- MC score conflates difficulty and policy uncertainty [2505.13427#s1]
- Filter degenerates with a thinking completer (>98% correct) [2506.09532#s1]
- Selection only; gains shrink for strong policies [2505.13427#s2]
- Multiple rollouts per step are costly and noisy [2606.11209#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2501.04686#s1` — Unlocking Multimodal Mathematical Reasoning via Process Reward Model (https://arxiv.org/html/2501.04686v6; Sec 3.1; Table 3; Sec 6.2)
- `2505.13427#s1` — MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision (https://arxiv.org/html/2505.13427v2; Sec 3.3; Sec 4.1; Table 1)
- `2505.13427#s2` — MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision (https://arxiv.org/html/2505.13427v2; Sec 3.4; Table 1; Table 2; Fig 3; Sec 5.1-5.3)
- `2506.09532#s1` — Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models (https://arxiv.org/html/2506.09532v5; Sec 3.2; Table 5; Appendix A.2 Table 6; A.8 Table 12)
- `2506.09532#s2` — Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models (https://arxiv.org/html/2506.09532v5; Sec 3.3; Table 5; Appendix A.5 Tables 9-10)
- `2606.11209#s1` — ProcessThinker: Enhancing Multi-modal Large Language Models Reasoning via Rollout-based Process Reward (https://arxiv.org/html/2606.11209v1; Table 1; Sec 2.2 Eq 2-6; Sec 3)
