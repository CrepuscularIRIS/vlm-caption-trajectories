# Train the step verifier to write a rationale or checklist before its verdict

Card `K23-generative-prm-with-rationale` · status: candidate_pattern · members: 2 signatures from 2 full-text papers

## Operation

Fine-tune the step verifier generatively to produce a checklist/rationale and then a verdict token, and read the step score from the verdict probability, using expert or human step labels.

- `2505.15277#s1`: fine-tune a small LM/VLM with next-token prediction to (1) generate a subgoal checklist from the instruction and (2) given checklist, observation and action, generate textual feedback followed by Yes/No/In-Progress judgments per checklist item; score a step as the average over items of the soft probability mass on Yes (plus partial credit for In Progress) read from the LM-head logits
- `2509.23738#s3`: collect binary step labels from humans (balanced 1:1 positive/negative), augment with GPT-4o chain-of-thought rationales, and SFT the PRM to generate the rationale then a True/False token

**Transfer notes:** Both members are web/GUI agents; transfer to image-QA reasoning steps needs step labels that observe perception.

## When it works

- Checklist + feedback PRM beats GPT-4o; generative objective beats a Bradley-Terry head out of distribution [2505.15277#s1]
- Human binary labels with rationales give 94.9% PRM accuracy vs 89.7% from GPT-4o labels [2509.23738#s3]

## When it fails

- Halving instructions or rejected actions sharply cuts accuracy; hallucinated checklist items [2505.15277#s1]
- Rationale gain is small (+0.2/+0.3) [2509.23738#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2505.15277#s1` — Web-Shepherd: Advancing PRMs for Reinforcing Web Agents (https://arxiv.org/html/2505.15277v2; Sec 5, Table 1, Fig 6, Sec 7.2 Fig 8)
- `2509.23738#s3` — GUI-Shepherd: Reliable Process Reward and Verification for Long-Sequence GUI Tasks (https://arxiv.org/html/2509.23738v1; Table 3; Table 4; Sec 3, 5.2; Appendix A.2, D)
