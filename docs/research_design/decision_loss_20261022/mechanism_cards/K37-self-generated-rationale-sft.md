# SFT on rationales the model generated and filtered itself

Card `K37-self-generated-rationale-sft` · status: candidate_pattern · members: 4 signatures from 3 full-text papers

## Operation

Let the model generate its own rationales (best-PRM correct sample, contrast-rectified rationale, reasoning text from an RL-trained model on other datasets) and SFT on them.

- `2506.09532#s3`: keep queries with 2-6 of 8 correct responses, drop incorrect and near-duplicate responses, select the highest-PRM-scoring correct response as the SFT target, and fine-tune the policy (reward-ranked fine-tuning)
- `2603.02556#s1`: generate a hinted coarse rationale for the target, prompt the same VLM to compare the target with its counterpart (summarise common visual patterns if answers match, describe fine-grained differences if they differ), then have a text-only LLM rewrite the coarse rationale treating the contrastive analysis as more reliable, and filter outputs by text matching before SFT
- `2603.02556#s3`: build VC-STaR training sets from only positive pairs, only negative pairs, or both, and compare finetuned models
- `2510.11369#s1`: train GRPO separately on a single dataset, use the resulting model to relabel images from every dataset with its quality-reasoning text, then SFT the RL model on the pooled image-text pairs, adding score supervision only from one dataset

**Transfer notes:** n/a (in-domain).

## When it works

- PRM-ranked rejection-sampled targets: MathVista 68.1 -> 71.4 [2506.09532#s3]
- Rationales rectified by contrast with a similar sample: avg 58.1 vs 55.5, above STaR [2603.02556#s1]
- Negative (different-answer) counterparts help most [2603.02556#s3]
- RL model relabels other datasets with reasoning text: OOD PLCC 0.858 vs 0.831 [2510.11369#s1]

## When it fails

- Baselines from leaderboards; MathVision +0.3 [2506.09532#s3]
- Rewriting LLM never sees the image; MMStar +0.6 [2603.02556#s1]
- Single dataset [2603.02556#s3]
- In-domain fit below SFT methods [2510.11369#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2506.09532#s3` — Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models (https://arxiv.org/html/2506.09532v5; Sec 3.5; Sec 4.3 Table 4; Appendix A.6 Table 11)
- `2603.02556#s1` — Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs (https://arxiv.org/html/2603.02556v1; Table 1; Table 2; Sec 4.2; Figure 1(b))
- `2603.02556#s3` — Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs (https://arxiv.org/html/2603.02556v1; Table 4; Sec 4.3)
- `2510.11369#s1` — Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment (https://arxiv.org/html/2510.11369v2; Sec 3.4; Table 3; Table 5; Appendix C Tables A.2-A.3)
