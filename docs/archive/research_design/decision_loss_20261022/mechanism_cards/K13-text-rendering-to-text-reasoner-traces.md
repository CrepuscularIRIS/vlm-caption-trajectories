# Write reasoning traces from a text rendering of the image with a text-only reasoner

Card `K13-text-rendering-to-text-reasoner-traces` · status: candidate_pattern · members: 2 signatures from 2 full-text papers

## Operation

Render the image as text (question-targeted MLLM description, privileged markdown table), have a text-only reasoning LLM write long CoT from it with GT-answer filtering, then SFT the MLLM on (image, trace).

- `2503.06749#s1`: convert visual content to a question-targeted textual description via an intermediate Pseudo-CoT, have a text-only reasoning LLM generate long reflective CoT from the text, keep traces whose final answer matches GT after rule-based logical-consistency filtering and word replacement, and pair the text CoT with the original image for SFT
- `2506.04088#s1`: Prompt a reasoning LLM with the structured table, question and GT answer to write an answer-justifying chain, reject-sample verbose/contradictory/incoherent traces with an auxiliary LLM judge, then SFT the MLLM on (table image, question) -> <think>trace</think><answer>

**Transfer notes:** Instance of brief mapping 2: the text rendering is training-only privileged information, and the failure mode (traces citing unreadable content) is the imitation gap.

## When it works

- Cold start beats LLaVA-CoT and Mulberry on Llama-3.2-11B-V [2503.06749#s1]
- Privileged markdown tables give SFT gains for table images [2506.04088#s1]

## When it fails

- Traces inherit description errors; cold start alone scores below the base with overthinking [2503.06749#s1]
- Traces may cite cell values the student cannot read from the image; trace-image consistency not measured [2506.04088#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2503.06749#s1` — Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models (https://arxiv.org/html/2503.06749v4; Sec 3.3.1; Table 2; Table 4; Table 6; Appendix A-B, Fig 5)
- `2506.04088#s1` — Multimodal Tabular Reasoning with Privileged Structured Information (https://arxiv.org/html/2506.04088v1; Sec 4.1; Sec 5.3 Ablation, Fig 4; Table 1)
