# Warm-start tool RL with synthesized, correctness-filtered multi-turn trajectories

Card `K12-tool-trajectory-cold-start` · status: candidate_pattern · members: 6 signatures from 6 full-text papers

## Operation

Synthesize multi-turn thought-action-observation trajectories (teacher agent loop, few-shot exemplars, MCTS, templates with inserted wrong crops) or SFT on the hard items themselves, filter by final-answer correctness, and SFT before tool RL.

- `2509.07969#s2`: Prompt an existing VLM few-shot with the exemplars to iteratively produce thought and crop action per turn on new hard queries, execute crops, and retain only trajectories ending in a correct answer; then SFT the base model on them
- `2602.14201#s1`: generate teacher trajectories where the teacher decides at each round to answer or issue a normalized-bbox zoom on any prior view, filter by answer agreement with the reference, strip non-informative turns, and SFT the student on the resulting mix of no-zoom, single-zoom and multi-zoom traces
- `2512.16918#s2`: Generate multi-turn tool-using CoT with a large teacher, rule-filter to ~100k traces, and SFT the 7B base for one epoch before RL
- `2505.15966#s2`: synthesize template-based trajectories (global analysis, operation on the reference cue, local analysis, answer) with GPT-4o, insert wrong crops/frames (disjoint, oversized boxes, wrong frames) before the correct one to form self-correction traces, mix in 2K text-only traces, and train with loss masked on execution outputs and on the inserted erroneous operations
- `2505.23678#s1`: Run MCTS (UCB selection, teacher expansion of grounded steps, rollout, answer-correctness backprop) over grounded reasoning steps, linearize root-to-leaf paths into direct-correct chains and failed-then-'wait, that seems off'-corrected chains, and SFT the base model on them before RL
- `2602.14225#s2`: pre-warm the policy by SFT on the same hard image-text examples that will later be optimised with tool-based GRPO, instead of introducing them only during RL

**Transfer notes:** n/a (in-domain).

## When it works

- Without cold start RL produces 1-turn trajectories (VisualProbe hard 25.4 vs 44.4) [2509.07969#s2]
- Domain-matched teacher zoom traces break the always-one-call pattern (XLRS 52.87 vs 47.73) [2602.14201#s1]
- Teacher tool traces then RL: 67.6 vs 65.9 for RL only [2512.16918#s2]
- Self-correction traces with inserted wrong crops are needed for RL to keep using tools [2505.15966#s2]
- MCTS-built grounded chains: without them GRPO never yields valid formatted outputs [2505.23678#s1]
- Pre-warming on the same tiny-target items RL will optimize lets RL find rewarded trajectories [2602.14225#s2]

## When it fails

- SFT induces tool overuse that RL must then reduce [2512.16918#s2]
- Warm-start alone underperforms the base model on several benchmarks [2505.15966#s2]
- General-domain tool traces do not transfer to UHR remote sensing [2602.14201#s1]
- Pattern diversity bounded by hand-crafted exemplars [2509.07969#s2]
- Coordinates unsupervised: 72.8% judged accurate [2505.23678#s1]
- No test of whether pre-warming on the same items inflates in-distribution gains [2602.14225#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2509.07969#s2` — Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search (https://arxiv.org/html/2509.07969v1; Sec 3.2; Table 2 (ID2 vs ID4))
- `2602.14201#s1` — GeoEyes: On-Demand Visual Focusing for Evidence-Grounded Understanding of Ultra-High-Resolution Remote Sensing Imagery (https://arxiv.org/html/2602.14201v2; Table 4; Table 8-9 (Appendix B); Sec 3)
- `2512.16918#s2` — AdaTooler-V: Adaptive Tool-Use for Images and Videos (https://arxiv.org/html/2512.16918v3; Sec 3.2 CoT Annotation; Sec 4.3.2; Table 3; Table 5)
- `2505.15966#s2` — Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning (https://arxiv.org/html/2505.15966v3; Sec 3; Sec 5.2; Table 1; Appendix C.2 Table 2; Appendix D.1)
- `2505.23678#s1` — Grounded Reinforcement Learning for Visual Reasoning (https://arxiv.org/html/2505.23678v3; Sec 4.2; Table 4; Table 5)
- `2602.14225#s2` — Text Before Vision: Staged Knowledge Injection Matters for Agentic RLVR in Ultra-High-Resolution Remote Sensing Understanding (https://arxiv.org/html/2602.14225v1; Sec 3.1-3.2; Table 1; Fig 3; Table 4)
