# Replace SFT with RL or preference training on the same perception data

Card `K31-swap-sft-for-rl-on-perception-data` · status: candidate_pattern · members: 3 signatures from 3 full-text papers

## Operation

Train on the same perception items with RL or DPO (on-policy or preference contrast) instead of supervised targets.

- `2510.16333#s1`: replace SFT on chosen responses with DPO on chosen/rejected pairs while letting gradients flow into the vision encoder
- `2605.20177#s3`: train perception with RLVR (GRPO, accuracy reward on short answers) instead of caption-based SFT before the reasoning stages
- `2503.01785#s2`: Sample a group of think-then-answer responses, reward 1/0 exact class match (or IoU for grounding boxes) plus format reward, update with GRPO; compare against SFT on the same few examples

**Transfer notes:** n/a (in-domain).

## When it works

- DPO beats SFT on vision-heavy VQA and improves encoder probes when the encoder is unfrozen [2510.16333#s1]
- RLVR perception stage beats caption-SFT: math avg 42.26 vs 37.03 (Qwen2.5-VL-7B) [2605.20177#s3]
- One-shot fine-grained classification 80.3 vs 51.7 for SFT [2503.01785#s2]

## When it fails

- Knowledge VQA gap marginal; supervision not equalized [2510.16333#s1]
- Small gap on Qwen3-VL-8B; SFT wins MathVision there [2605.20177#s3]
- Effect of the think segment not separated; smaller at 7B [2503.01785#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2510.16333#s1` — RL makes MLLMs see better than SFT (https://arxiv.org/html/2510.16333v2; Sec 3.2 Figs 2-4; Sec 4.2 Figs 5-8, 10; Sec 5 Table 1; Appendix B.1, B.4)
- `2605.20177#s3` — From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models (https://arxiv.org/html/2605.20177v1; Sec 4.4; Table 4)
- `2503.01785#s2` — Visual-RFT: Visual Reinforcement Fine-Tuning (https://arxiv.org/html/2503.01785v1; Sec 3.2.1 CLS reward; Table 2; Table 6; Sec 4.2, 4.4)
