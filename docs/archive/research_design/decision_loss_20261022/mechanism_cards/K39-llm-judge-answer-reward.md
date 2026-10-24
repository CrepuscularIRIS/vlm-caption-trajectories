# Use a text LLM judge against the gold answer as the RL reward

Card `K39-llm-judge-answer-reward` · status: candidate_pattern · members: 2 signatures from 2 full-text papers

## Operation

Let a text LLM compare the final answer to the gold answer (binary verdict or a fixed step-by-step protocol) to reward open-ended answers; the judge does not see the image.

- `2507.13348#s2`: use an LLM judge to output a discrete 0/1 correctness reward for GRPO on general VQA data (no rule-based checker)
- `2605.14054#s3`: instruct the LLM judge to execute a fixed universal step-by-step verification protocol (classify answer type, extract boxed content, type-specific comparison with partial-credit table) instead of a holistic equivalence judgment

**Transfer notes:** n/a (in-domain).

## When it works

- Binary LLM-judge reward improves general VQA [2507.13348#s2]
- Structured verification protocol: 91.9% accuracy vs 79.1% for plain prompting [2605.14054#s3]

## When it fails

- Judge reliability unmeasured; cannot detect correct guesses without visual grounding [2507.13348#s2]
- Judge stochastic; a plain LLM judge is exploited [2605.14054#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2507.13348#s2` — VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning (https://arxiv.org/html/2507.13348v1; Sec 3.2; Table 1; Appx B.5 Table 6)
- `2605.14054#s3` — Bad Seeing or Bad Thinking? Rewarding Perception for Multimodal Reasoning (https://arxiv.org/html/2605.14054v2; Sec 2.4; Table 4; Table 2 (w/o SVV+PV); Appendix D; Appendix E.1)
