# Train perception tasks with direct answers instead of think-then-answer

Card `K09-drop-cot-for-perception-tasks` · status: candidate_pattern · members: 3 signatures from 3 full-text papers

## Operation

For grounding, detection, OCR and segmentation, use an answer-only format reward/template rather than requiring a reasoning segment.

- `2504.07954#s2`: train and infer with direct answer output (no explicit CoT) under GRPO, and choose RL over SFT according to the task's perceptual perplexity (multi-object counting/detection) rather than uniformly
- `2505.15810#s3`: train with a Fast Thinking Template that asks directly for bbox JSON instead of a think-then-answer template
- `2510.21978#s2`: use an answer-only format reward for perception domains (segmentation, grounding) and keep the thinking-format reward only for reasoning domains

**Transfer notes:** n/a. Holds where the answer is read off the image (coordinates, short strings); untested for multi-step visual reasoning.

## When it works

- RefCOCO 75.1 with thinking vs 89.1 without; PageOCR 77.3 vs 95.7 [2504.07954#s2]
- Fast template: 90.3% ScreenSpot with about one third of the tokens [2505.15810#s3]
- Under a uniform thinking reward the model shrinks segmentation CoT to near zero anyway [2510.21978#s2]

## When it fails

- 'Perceptual perplexity' is asserted, not measured; 2B models may reason weakly [2504.07954#s2]
- No isolated ablation; length-accuracy relation is correlational [2505.15810#s3]
- No ablation of the domain-specific format reward [2510.21978#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2504.07954#s2` — Perception-R1: Pioneering Perception Policy with Reinforcement Learning (https://arxiv.org/html/2504.07954v1; Table 5 rows 3-8; Sec 5.2)
- `2505.15810#s3` — GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents (https://arxiv.org/html/2505.15810v2; Sec 3.1, Figure 2, Table 4, Appendix D.2 Table 7)
- `2510.21978#s2` — Beyond Reasoning Gains: Mitigating General-Capability Forgetting in Large Reasoning Models (https://arxiv.org/html/2510.21978v2; Sec 5.3 Fig 6)
