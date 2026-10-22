# Create labelled negatives by minimally editing verified-correct content

Card `K20-synthetic-negatives-by-gold-edit` · status: candidate_pattern · members: 5 signatures from 5 full-text papers

## Operation

Make negatives whose label is known by construction through minimal controlled edits of correct captions, solutions, responses, prompts, or inpainted image regions, and train reward models, verifiers, judges or preference learners on them.

- `2410.04734#s1`: prompt a text-only LLM to make a minimal-edit perturbation of a gold caption/answer in one of eight taxonomies (spatial relation, attribute, binding, object identity, counting, small object, OCR, counterfactual), filter with an LLM checker, and label tokens that differ from the original as negative
- `2501.04686#s2`: have the external model caption the image, replace a confusable visual condition in a correct solution with an adjacent or similar wrong one, continue reasoning from it, and label every step from the insertion onward as incorrect
- `2407.02477#s3`: replace each original rejected response with a GPT-4-corrupted copy of the chosen response containing injected realistic hallucinations, then subsample 5k pairs and train with DPO
- `2510.13804#s1`: construct counterfactual false examples in reverse from a faithful description, either by editing the prompt with the image fixed or by inpainting a chosen object with the prompt fixed, and keep the pair with a known label and explanation
- `2606.02578#s1`: edit one or a few extracted visual attributes (object count, color, spatial relation, presence of an entity) in the correct response while keeping its structure and reasoning, producing r_p; additionally insert a reasoning error that depends on the altered attribute to produce r_p+r; discard candidates whose final answer does not diverge from ground truth

**Transfer notes:** n/a (in-domain).

## When it works

- Text-only minimal-edit perturbations give token labels without human annotation [2410.04734#s1]
- Inserted confusable visual conditions add PRM data (removal lowers BoN) [2501.04686#s2]
- 5k GPT-4-corrupted copies of chosen responses beat 80k original VLFeedback pairs on POPE [2407.02477#s3]
- Prompt edits and inpainted objects with known labels train a verifier above GPT-4o [2510.13804#s1]
- Attribute edits of correct responses give a slightly better judge dataset [2606.02578#s1]

## When it fails

- Edits are image-blind; mAP on negatives only ~41-46% [2410.04734#s1]
- Synthetic errors may not match the policy's natural perception errors [2501.04686#s2]
- MMVet regresses [2407.02477#s3]
- Falls below base on some verifier tasks [2510.13804#s1]
- Gain <=0.004 per metric in the controlled ablation [2606.02578#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2410.04734#s1` — TLDR: Token-Level Detective Reward Model for Large Vision Language Models (https://arxiv.org/html/2410.04734v2; Sec 4; Sec 5.1 human evaluation paragraph)
- `2501.04686#s2` — Unlocking Multimodal Mathematical Reasoning via Process Reward Model (https://arxiv.org/html/2501.04686v6; Sec 3.2; Table 3; Sec 6.2)
- `2407.02477#s3` — Understanding Alignment in Multimodal LLMs: A Comprehensive Study (https://arxiv.org/html/2407.02477v1; Sec 5.3, Table 5, Table 2)
- `2510.13804#s1` — Generative Universal Verifier as Multimodal Meta-Reasoner (https://arxiv.org/html/2510.13804v1; Sec 4.1; Sec 4.3; Tables 1-2)
- `2606.02578#s1` — Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling (https://arxiv.org/html/2606.02578v1; Sec 3.3; Appendix D.2 Steps 1-4; Table 3 rows 2-3)
