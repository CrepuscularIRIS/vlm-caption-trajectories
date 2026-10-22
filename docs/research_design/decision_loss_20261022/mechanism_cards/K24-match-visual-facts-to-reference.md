# Score a response's stated visual facts against a reference fact list

Card `K24-match-visual-facts-to-reference` · status: candidate_pattern · members: 5 signatures from 3 full-text papers

## Operation

Extract reference visual facts (from a stronger model's correct CoT, a distilled constraint extractor, or annotated scene components) and reward or gate the response by how well its stated visual content matches them.

- `2506.07218#s1`: Have a judging LLM mark each reference visual annotation as present/absent in the rollout, use the present fraction as a perception reward r_v, and add gamma*r_v plus an n-gram repetition penalty to format and accuracy rewards before GRPO advantage normalization
- `2506.07218#s2`: Replace the annotation-matching judge with a direct image-response consistency score from a larger MLLM
- `2603.16253#s1`: match each checklist claim against the predicted constraints with type-specific rules (numeric tolerance 15%, Jaccard entity overlap, confidence-weighted; unmatched well-formed claim = 0, unparsable = 0.5), aggregate by smoothed geometric mean into r, and multiply rewards of visually dependent steps by sigmoid(beta(r - tau)) while leaving non-visual steps unchanged
- `2603.16253#s2`: distil solution-critical constraint JSON from a large teacher given reference solutions into a 7B extractor by SFT on 26,454 pairs, then DPO on 5,832 teacher-rewritten preferred/rejected constraint pairs
- `2510.04532#s2`: apply GRPO with these rewards to SFT and base models, then test with and without priors and under prior perturbation

**Transfer notes:** n/a (in-domain).

## When it works

- Fraction of reference annotations present in the rollout as reward: MathVerse 51.3 -> 54.3 [2506.07218#s1]
- Gating visually dependent step rewards by premise-constraint match: VisualProcessBench Macro-F1 +7 for a large judge [2603.16253#s1]
- A 7B extractor distilled from a teacher with reference solutions reaches F1 0.87 [2603.16253#s2]

## When it fails

- Weak judges make the perception reward saturate early (hacked) [2506.07218#s1]
- Gain mixes extra evidence with gating; one misread propagates to the whole trajectory [2603.16253#s1]
- Relation constraints least reliable [2603.16253#s2]

## Counterexamples

- A direct MLLM image-response consistency reward saturates by ~100 steps and is hacked [2506.07218#s2]
- Rewarding mention of GT key visual components reaches >0.7 matching but the plan stays unchanged [2510.04532#s2]

## Sources

- `2506.07218#s1` — Perception-R1: Advancing Multimodal Reasoning Capabilities of MLLMs via Visual Perception Reward (https://arxiv.org/html/2506.07218v3; Sec 4.2; Table 1; Table 2; Fig 3(a); Sec 5.2 McNemar)
- `2506.07218#s2` — Perception-R1: Advancing Multimodal Reasoning Capabilities of MLLMs via Visual Perception Reward (https://arxiv.org/html/2506.07218v3; Table 2 (Approach Ablation); Appendix B.6, Fig 5)
- `2603.16253#s1` — Grounding the Score: Explicit Visual Premise Verification for Reliable Vision-Language Process Reward Models (https://arxiv.org/html/2603.16253v2; Table 1; Table 6; Sec 3.2-3.4; Appendix B, F)
- `2603.16253#s2` — Grounding the Score: Explicit Visual Premise Verification for Reliable Vision-Language Process Reward Models (https://arxiv.org/html/2603.16253v2; Sec 3.5; Table 4; Figure 4; Appendix C.1-C.2, G Table 11)
- `2510.04532#s2` — More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models (https://arxiv.org/html/2510.04532v2; Table 2; Sec 3.2, 4.2; Appendix A.7, A.15.2 Table 12)
