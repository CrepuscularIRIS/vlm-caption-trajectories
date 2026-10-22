# Learn the new task from the model's own samples to reduce forgetting

Card `K30-on-policy-data-forgets-less` · status: candidate_pattern · members: 3 signatures from 2 full-text papers

## Operation

Train on data from the model's own distribution (sequential GRPO, or SFT on self-generated correct rollouts) rather than external targets, using base-model perplexity as the predictor of interference.

- `2507.05386#s1`: Replace sequential SFT with sequential GRPO (or RLOO/ReMax) on each new task, using group-normalized rewards so prompts with identical rewards across rollouts produce zero advantage and no update
- `2506.23508#s1`: Sample reasoning rollouts from the model's own RL policy on the new-task training set, keep only those whose final answer is correct (~65% kept), and run standard SFT of the base model on this self-generated corpus with the same hyperparameters as other SFT arms
- `2506.23508#s2`: Estimate interference via a lower bound of the empirical NTK norm (LBK) between training data and prior-knowledge samples, and measure base-model perplexity of candidate training data to predict which corpus will cause forgetting

**Transfer notes:** n/a (in-domain); a data-distribution alternative to the parameter-side preservation of K26.

## When it works

- Sequential GRPO forgets far less than sequential SFT (FM -2.3 vs -10.4) [2507.05386#s1]
- SFT on own correct rollouts: RefCOCO drop 4.2 vs 14.6 (GPT-4o rationales) and 82.8 (answer-only) at 3B [2506.23508#s1]
- Low-perplexity data and interference bounds match the forgetting order [2506.23508#s2]

## When it fails

- Lower plasticity on some tasks (PathVQA 41.3 vs 62.2); no-KL GRPO unstable [2507.05386#s1]
- One novel task; forgetting not zero [2506.23508#s1]
- Interference magnitude alone mispredicts [2506.23508#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2507.05386#s1` — Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training (https://arxiv.org/html/2507.05386v6; Table 1-8; Sec 4.2-5.1)
- `2506.23508#s1` — Why Reinforcement Fine-Tuning Enables MLLMs Preserve Prior Knowledge Better: A Data Perspective (https://arxiv.org/html/2506.23508v4; Table 1; Table 5; Sec 5.3, 5.8; Table 2 (LLM math replication))
- `2506.23508#s2` — Why Reinforcement Fine-Tuning Enables MLLMs Preserve Prior Knowledge Better: A Data Perspective (https://arxiv.org/html/2506.23508v4; Theorems 5.1-5.2; Fig 2-4, 6-7; Sec 5.4-5.5)
