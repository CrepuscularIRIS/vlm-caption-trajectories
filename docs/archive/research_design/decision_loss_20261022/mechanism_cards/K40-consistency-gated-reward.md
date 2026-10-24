# Gate the reward on the internal consistency of the trajectory

Card `K40-consistency-gated-reward` · status: candidate_pattern · members: 4 signatures from 4 full-text papers

## Operation

Gate or add reward by the internal consistency of the rollout (CoT conclusion vs answer, need for zoom vs zoom made, textual CoT vs a stored grounded reference, thought vs action arguments), checked by a judge or rules without visual access.

- `2602.12506#s3`: Grant correctness reward only when the chain of thought and final answer are judged consistent
- `2602.14201#s3`: score each rollout with a 'necessity-aware' process judge that, beyond checking reasoning-answer consistency, penalizes confident answers to detail-oriented queries produced without a corresponding zoom-in
- `2605.31096#s1`: in parallel with a box-grounded rollout group, sample a textual rollout group and reward each textual CoT by an LLM-judged 4-level consistency score (1/0.7/0.3/0) against the best correct grounded rollout with box reward > tau, kept in a per-query rollout archive
- `2509.24304#s2`: terminate any trajectory that fails redundancy, logical-flow or fidelity checks and multiply its reward by 0 (CCV); at inference terminate and fall back to a direct-answer prompt

**Transfer notes:** n/a (in-domain).

## When it works

- Consistency gate keeps accuracy and faithfulness aligned without augmentation [2602.12506#s3]
- Necessity-aware judge: 54.23 vs 52.82 with logic-only judge [2602.14201#s3]
- Textual CoT rewarded by consistency to the best grounded rollout: 72.4 vs 69.1 textual-only [2605.31096#s1]
- Rule checks terminate redundant or inconsistent actions [2509.24304#s2]

## When it fails

- Judge without visual access rewards the presence of a zoom, not its correctness [2602.14201#s3]
- Judge hallucination corrupts reward; needs GT boxes; 11-23% of queries have no reference [2605.31096#s1]
- Only syntactic/numeric consistency [2509.24304#s2]

## Counterexamples

- With caption/CoT augmentation the consistency-gated reward collapses into copying the provided text [2602.12506#s3]

## Sources

- `2602.12506#s3` — On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs (https://arxiv.org/html/2602.12506v3; Sec 3.3; Appendix F.3; Fig 23; Boxes 25-26)
- `2602.14201#s3` — GeoEyes: On-Demand Visual Focusing for Evidence-Grounded Understanding of Ultra-High-Resolution Remote Sensing Imagery (https://arxiv.org/html/2602.14201v2; Sec 4.2.2; Table 6; Table 7)
- `2605.31096#s1` — iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning (https://arxiv.org/html/2605.31096v1; Sec 3.2-3.3; Table 2; Table 5; Tables 6-8; Table 10)
- `2509.24304#s2` — FrameThinker: Learning to Think with Long Videos via Multi-Turn Frame Spotlighting (https://arxiv.org/html/2509.24304v2; Sec 3.3.2 Eq 6; Fig 4(c); Appendix D.3, B.2)
