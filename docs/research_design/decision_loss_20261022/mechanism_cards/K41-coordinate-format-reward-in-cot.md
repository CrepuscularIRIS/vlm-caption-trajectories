# Reward valid coordinates inside the reasoning without checking what they point at

Card `K41-coordinate-format-reward-in-cot` · status: candidate_pattern · members: 3 signatures from 2 full-text papers

## Operation

Require or reward syntactically valid boxes/points (and box counts equal to the GT count) inside the reasoning trace, with only answer correctness as the semantic signal.

- `2505.15879#s1`: reward the presence of regex-valid bounding-box coordinates inside the think segment plus correct think/rethink token structure, combined with a GPT-4o-judged answer-accuracy reward (plus down-weighted BLEU-1), and optimize with GRPO
- `2505.15879#s2`: give a reward when the number of generated bounding boxes in the reasoning exactly equals the GT object count
- `2505.23678#s2`: Give format reward only when think/answer tags are correct and every coordinate reference is valid, add task-correctness reward, and optimize with GRPO from the grounded warm-start

**Transfer notes:** n/a (in-domain). Contrast with K07, which uses GT locations.

## When it works

- 20 training examples: VSR 72.9 vs 59.7 few-shot SFT [2505.15879#s1]
- Box-count reward on counting items raises grounding IoU [2505.15879#s2]
- Grounded warm start + valid-coordinate format reward: SAT-2 62.9 vs 50.0 vanilla GRPO [2505.23678#s2]

## When it fails

- Box semantic accuracy never rewarded; MathVista gains small or negative [2505.15879#s1]
- Ablation confounds data and reward; box count can disagree with text [2505.15879#s2]
- Reward hacking with partially incorrect reasoning [2505.23678#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2505.15879#s1` — GRIT: Teaching MLLMs to Think with Images (https://arxiv.org/html/2505.15879v2; Sec 3.2; Table 1; Figure 4-5; Sec 4.3)
- `2505.15879#s2` — GRIT: Teaching MLLMs to Think with Images (https://arxiv.org/html/2505.15879v2; Appendix B/C, Table 3)
- `2505.23678#s2` — Grounded Reinforcement Learning for Visual Reasoning (https://arxiv.org/html/2505.23678v3; Sec 4.3; Tables 1, 2, 4)
