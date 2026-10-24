# Reward or penalise tool calls by the item's estimated tool benefit

Card `K11-benefit-conditioned-tool-reward` · status: candidate_pattern · members: 4 signatures from 3 full-text papers

## Operation

Estimate per item whether the tool or high resolution helps (base or reference model rollouts with vs without it, or in-group statistics) and set the sign and size of the tool reward from that benefit and the call count.

- `2507.13348#s1`: add a control penalty 0.1*[1_direct*I(r<theta) + 1_high*I(r>=theta)] with r=C_direct/(C_direct+C_high), theta=0.2, so direct answers are penalized when low-res rarely succeeds and high-res requests are penalized when it often does
- `2507.13348#s3`: label samples as low-res-solvable if both resolutions are correct in all 8 rollouts, and high-res-needed if high-res correct count exceeds low-res by >=6; select 10K of each
- `2512.16918#s1`: Add a tool reward alpha * f(dS, n) to the correctness+format reward, where the sign follows the sample's Tool Benefit Score dS and the magnitude follows a Gaussian-decay function of the number of tool calls n relative to a maximum (sigma=2), so tool calls are rewarded on tool-helpful samples and increasingly penalized on tool-unhelpful ones
- `2602.14201#s2`: add to the format-gated accuracy reward an efficiency term R_tool = (1 - p_base(y|x)) * exp(-gamma * max(0, N_step - N_base(C))), so steps beyond the category allowance decay the bonus and easy-for-base items get little tool bonus; and a coarse-to-fine term that gives +beta_zoom when the next box is strictly contained in and smaller than the previous, 0 for backtracking to a containing box, and -beta_drift for disjoint/other moves

**Transfer notes:** Brief mapping 1 (next-best-view) is the cross-domain analogue (K54), but here benefit is an accuracy delta, not information gain under a belief. Requires verifiable answers and an extra offline rollout pass.

## When it works

- Group-ratio control penalty avoids both always-request and never-request collapse; about 51% of tokens [2507.13348#s1]
- Training data split into low-res-solvable and high-res-needed by base rollouts [2507.13348#s3]
- Tool Benefit Score from a reference model signs the tool reward: avg 67.6 -> 69.9 [2512.16918#s1]
- Efficiency term scaled by base-model failure probability and step allowance: removing it costs about 2 points [2602.14201#s2]

## When it fails

- Raising the threshold shifts the policy to always requesting high-res; custom prompts collapse to direct answers [2507.13348#s1]
- Benefit from a single reference model may be biased; small alpha lowers performance [2512.16918#s1]
- Removing the tool term leaves the call ratio unchanged, so its effect is not through call frequency; containment reward rewards narrowing regardless of target [2602.14201#s2]
- Selection rule not ablated [2507.13348#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2507.13348#s1` — VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning (https://arxiv.org/html/2507.13348v1; Sec 3.4 Eq. 5; Fig 3(a); Fig 5; Table 2; Appx C.4 Fig 8)
- `2507.13348#s3` — VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning (https://arxiv.org/html/2507.13348v1; Sec 3.5)
- `2512.16918#s1` — AdaTooler-V: Adaptive Tool-Use for Images and Videos (https://arxiv.org/html/2512.16918v3; Sec 3.3 Eqs. 1-3; Table 3; Table 4; Fig 5)
- `2602.14201#s2` — GeoEyes: On-Demand Visual Focusing for Evidence-Grounded Understanding of Ultra-High-Resolution Remote Sensing Imagery (https://arxiv.org/html/2602.14201v2; Sec 4.2.2 Eq 1-2; Tables 5, 7, 8, 9; Appendix B)
