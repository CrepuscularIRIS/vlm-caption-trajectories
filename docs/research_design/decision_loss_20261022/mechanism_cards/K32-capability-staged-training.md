# Order training stages by capability

Card `K32-capability-staged-training` · status: candidate_pattern · members: 4 signatures from 4 full-text papers

## Operation

Sequence the training stages by capability (perception before visual reasoning, policy RL before critic RL, an alignment phase before hard-text mixing, domain text knowledge before agentic RL) instead of shuffling all data.

- `2605.20177#s2`: run GRPO sequentially perception -> text reasoning -> visual reasoning (equal epochs per stage) instead of shuffling all data together; optionally sort easy-to-hard by pass@16 within each stage
- `2509.00676#s3`: Train policy RL first, then continue GRPO on critic data (policy-then-critic), rather than mixing or critic-then-policy
- `2501.02669#s2`: first train a short alignment phase on simple Text and Image-via-Text examples (1e4), then train on an equal mix of hard Text supervision and simple Mix supervision (Align-Mix+)
- `2602.14225#s1`: SFT the VLM for one epoch on domain text-only QA with CoT, mixed with the hard UHR image-text VQA set, before running zoom-in-tool GRPO

**Transfer notes:** n/a (in-domain).

## When it works

- Perception -> text reasoning -> visual reasoning beats merged GRPO (59.75 vs 58.34); reverse order hurts [2605.20177#s2]
- Policy-then-critic data beats mixing and critic-then-policy [2509.00676#s3]
- Short alignment phase then hard-text + simple-mix: hard-image 76/96/56% [2501.02669#s2]
- Domain text QA with CoT before zoom RL: pass@1 52.4 -> 60.4 [2602.14225#s1]

## When it fails

- Another order is equally good; encoder-freezing confound; 1-2 points [2605.20177#s2]
- Critic ability still below critic-only [2509.00676#s3]
- Synthetic tasks only; Visual Analogy 56% [2501.02669#s2]
- Text alone insufficient; single benchmark [2602.14225#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2605.20177#s2` — From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models (https://arxiv.org/html/2605.20177v1; Sec 3.2; Sec 4.3.1-4.3.2; Sec 4.5; Tables 2, 3, 5, 7, 8)
- `2509.00676#s3` — LLaVA-Critic-R1: Your Critic Model is Secretly a Strong Policy Model (https://arxiv.org/html/2509.00676v1; Sec 4.2; Table 7)
- `2501.02669#s2` — Generalizing from SIMPLE to HARD Visual Reasoning: Can We Mitigate Modality Imbalance in VLMs? (https://arxiv.org/html/2501.02669v2; Sec 4.1-4.3; Figures 5-6)
- `2602.14225#s1` — Text Before Vision: Staged Knowledge Injection Matters for Agentic RLVR in Ultra-High-Resolution Remote Sensing Understanding (https://arxiv.org/html/2602.14225v1; Sec 3.3; Sec 5.1-5.2; Table 3; Table 4; Appendix C)
