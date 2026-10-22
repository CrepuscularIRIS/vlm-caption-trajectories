# Take answering away from the searching policy and let a frozen inspector gate termination

Card `K57-decouple-answer-authority` · status: case · members: 1 signatures from 1 full-text papers

## Operation

Remove the answer action from the planner; a frozen MLLM inspector sees only the submitted span and returns a sufficiency verdict that decides termination and answering; train only the planner.

- `2605.12571#s1`: remove the answer action from the planner policy; after each span submission, have a frozen MLLM inspector return a binary sufficiency verdict with feedback, terminate and answer only when the inspector says sufficient, otherwise feed the verdict back for further retrieval; train only the planner with GRPO

**Transfer notes:** n/a.

## When it works

- LVBench 48.2 -> 54.1 with lower temporal hallucination [2605.12571#s1]

## When it fails

- Inspector still answers 24% of non-target clips; misjudged sufficiency; latency [2605.12571#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2605.12571#s1` — VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority (https://arxiv.org/html/2605.12571v1; Table 1; Table 2; Table 3; Table 4; Sec 3.4, 4.1-4.3, 5.2-5.3; Appendix C.8)
