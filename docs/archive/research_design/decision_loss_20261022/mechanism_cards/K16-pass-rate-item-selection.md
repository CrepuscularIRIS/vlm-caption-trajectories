# Select training items by the model's own success rate

Card `K16-pass-rate-item-selection` · status: candidate_pattern · members: 4 signatures from 4 full-text papers

## Operation

Use base or current model rollouts to drop always-solved and never-solved items, keep items with some success, keep 'median' items, or keep items failed from the image but solved from a caption.

- `2505.14362#s2`: Filter out items the base model solves always (100%) or never (0%), convert to open-ended form, verify labels, keep fine-grained items only if solvable via cropping the GT region (perception-utility filter), then mix with chart and reasoning data
- `2507.05386#s2`: Roll out the current policy K times per instance and keep only instances with at least one rollout reward above a threshold (default: any non-zero reward), then run standard RFT on the filtered set
- `2603.02556#s2`: keep only 'median' samples that the base VLM answers wrongly alone but correctly when contrasting with the counterpart given the hint; drop easy (already correct) and hard (still wrong with contrast) samples
- `2605.20177#s1`: generate perception questions from captions, then keep an item only if the base VLM answers wrong from the image but right from the caption (checked with Qwen2.5-VL-7B and -32B)

**Transfer notes:** n/a (in-domain).

## When it works

- Filtering fine-grained data by pass rate and crop-solvability: V* 86.9 -> 91.6 [2505.14362#s2]
- Keeping items with any non-zero reward cuts GRPO time by 44% while still forgetting less than SFT [2507.05386#s2]
- Adding easy samples lowers Hallusion, MathVision, MMStar [2603.02556#s2]
- Items wrong from image but right from caption isolate perception: WeMath +7.43 [2605.20177#s1]

## When it fails

- Lower accuracy and more forgetting than full data; discards items that become learnable later; hard-only training degrades [2507.05386#s2]
- Fine-grained-only training degrades math reasoning [2505.14362#s2]
- Hard-sample inclusion not tested [2603.02556#s2]
- Needs fine-grained captions [2605.20177#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2505.14362#s2` — DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning (https://arxiv.org/html/2505.14362v3; Sec 3.3; Appendix B.1; Appendix B.2 Table 10)
- `2507.05386#s2` — Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training (https://arxiv.org/html/2507.05386v6; Algorithm 1; Table 9-12; Sec 5.2-5.3; Appendix B)
- `2603.02556#s2` — Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs (https://arxiv.org/html/2603.02556v1; Sec 3.1 Difficulty-Based Data Sampling; Table 3; Appendix A.2)
- `2605.20177#s1` — From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models (https://arxiv.org/html/2605.20177v1; Sec 3.1.1; Sec 4.2; Fig 3; Table 9)
