# Perturb inputs in ways whose effect on the answer is known

Card `K33-evidence-aware-input-perturbation` · status: candidate_pattern · members: 3 signatures from 3 full-text papers

## Operation

Augment training inputs with controlled perturbations: keep the label when the evidence is preserved (distractor concatenation, light blending, small rotation, correct or wrong text prefixes) or change the target to 'Unknown' when the evidence is masked.

- `2506.09736#s1`: Randomly apply one of: horizontally concatenate an irrelevant distractor image, alpha-blend a distractor with a skewed weight keeping the original dominant, or rotate by a small random angle, while keeping question and answer fixed; add perturbed copies alongside clean data (best: clean + 1x perturbed)
- `2509.20912#s1`: From each example build three variants that differ only in evidence availability: original image with GT answer; image with evidence regions masked with target 'Unknown'; image with irrelevant regions masked with the GT answer; train on all three so abstention is triggered by missing evidence rather than by the presence of masks
- `2602.12506#s2`: With 10% probability each, prepend a correct caption, wrong caption, correct thinking prefix, or wrong thinking prefix to a training question before sampling rollouts (60% left unmodified)

**Transfer notes:** n/a (in-domain).

## When it works

- Clean + 1x perturbed copies give 1-2 points across SFT, DPO and GRPO [2506.09736#s1]
- Evidence-masked variants with 'Unknown' targets: abstention 64% vs 46% (GRIT) and 33% (DeepEyes) [2509.20912#s1]
- Mixing correct and wrong caption/CoT prefixes restores Wrong-Caption accuracy [2602.12506#s2]

## When it fails

- Information-degrading perturbations (blur, crop, 50/50 mixup) hurt; rotation hurts OCR and charts [2506.09736#s1]
- Relies on detector/OCR evidence boxes [2509.20912#s1]
- Large seed variability; faithfulness still drifts [2602.12506#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2506.09736#s1` — Revisiting Visual Understanding in Multimodal Reasoning through a Lens of Image Perturbation (https://arxiv.org/html/2506.09736v2; Sec 4; Tables 2-5)
- `2509.20912#s1` — DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning (https://arxiv.org/html/2509.20912v4; Sec 3.2; Table 4 rows (1)-(2); Table 5)
- `2602.12506#s2` — On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs (https://arxiv.org/html/2602.12506v3; Sec 3.1-3.3; Fig 8-10)
