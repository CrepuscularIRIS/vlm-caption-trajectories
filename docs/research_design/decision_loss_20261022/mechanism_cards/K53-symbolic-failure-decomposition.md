# Decompose multimodal failures with symbolic ground-truth states

Card `K53-symbolic-failure-decomposition` · status: case · members: 2 signatures from 1 full-text papers

## Operation

Render symbolic problems into aligned diagram, text and per-atom probes; classify each failure as perception, reasoning or multimodal-specific; measure consistency across answer-preserving re-renders.

- `2510.01719#s1`: render each symbolic problem into an aligned diagram, a definitionally equivalent text description and one multiple-choice probe per atom, then classify each failure as perception (probe fails, text reasoning passes), reasoning (probes pass, text fails), both, or multimodal-specific (probes and text pass but diagram task fails)
- `2510.01719#s2`: apply answer-preserving transformations to the symbolic state, re-render diagrams, and compute Consistency Rate (expected pairwise agreement of predictions across variants)

**Transfer notes:** n/a (geometry-specific diagnostic).

## When it works

- RL reduces perception and reasoning errors together; the residual moves to multimodal-specific [2510.01719#s1]
- RL raises cross-variant consistency, SFT lowers it [2510.01719#s2]

## When it fails

- Multimodal-specific is a residual; checkpoints not retrained under control [2510.01719#s1]
- Geometry only; figures only [2510.01719#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2510.01719#s1` — What MLLMs Learn about When they Learn about Multimodal Reasoning (https://arxiv.org/html/2510.01719v4; Sec 2.1-2.2; Sec 3.2, 3.4; Fig 1, 6, 14)
- `2510.01719#s2` — What MLLMs Learn about When they Learn about Multimodal Reasoning (https://arxiv.org/html/2510.01719v4; Sec 3.5; Fig 9-10; Appendix F.2)
