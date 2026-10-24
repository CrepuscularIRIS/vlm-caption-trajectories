# Weight or select gradient tokens by their image-vs-perturbed-image divergence

Card `K17-token-visual-dependency-weighting` · status: candidate_pattern · members: 5 signatures from 3 full-text papers

## Operation

Compute per-token divergence between next-token distributions with the real image and with a masked/removed image (from the policy or a teacher), and use it to select, reweight or shape tokens' or trajectories' gradients.

- `2510.09285#s1`: Score every token by KL(pi(.|real image) || pi(.|perturbed image)), keep the top-k% (k=0.4) tokens per trajectory, and zero the policy-gradient contribution of all other tokens via a binary mask
- `2510.09285#s2`: Normalize each trajectory's mean visual dependency to a shaping factor in a range [beta_min, beta_max] (dynamic upper bound per batch) and multiply the trajectory's advantage by it
- `2603.25077#s1`: mark the top-alpha_r fraction of batch tokens by entropy as reasoning tokens and the top-alpha_p fraction by absolute image/no-image log-prob difference as perception tokens; multiply each token's clipped surrogate term by gamma_r (1.0) if reasoning or overlapping, gamma_p (0.5) if perception-only, and 0 otherwise
- `2603.25077#s2`: mask gradients on all tokens except the top-alpha fraction by entropy (reasoning-only) or by visual sensitivity (perception-only), sweeping alpha from 20% to 80%
- `2606.00564#s2`: for tokens with VDS above the 70th percentile, add lambda * KL(p_S(.|x) || q_T(.|x)) (text-only language-prior matching) to the steered loss; optionally (App E) set the per-token steering coefficient gamma_t to 0 / gamma/2 / gamma by VDS percentile bins 40/70

**Transfer notes:** n/a (in-domain).

## When it works

- Top-40% visually dependent tokens only: 55.0 -> 57.1 over DAPO, above entropy-based selection (56.0) [2510.09285#s1]
- Trajectory advantage scaled by mean visual dependency: +1.3 [2510.09285#s2]
- Reasoning-token (entropy) plus perception-token (image/no-image) reweighting: +1-3 points on math/hallucination [2603.25077#s1]
- Teacher visual-dependency score gates an extra language-prior term in distillation [2606.00564#s2]

## When it fails

- Too few or too many tokens both hurt; about 10% extra training time [2510.09285#s1]
- Down-weight factor outside about 0.3-0.9 falls below GRPO; tuned on the evaluation benchmarks [2603.25077#s1]
- The score measures reliance on the image, not whether the reliance is correct [2510.09285#s2]
- Single student per hyperparameter [2606.00564#s2]

## Counterexamples

- Training only on perception-sensitive tokens underperforms full-token GRPO on MathVerse (38.6-45.8 vs 50.8) [2603.25077#s2]

## Sources

- `2510.09285#s1` — Spotlight on Token Perception for Multimodal Reinforcement Learning (https://arxiv.org/html/2510.09285v2; Sec 3.2.3 Eq. 4; Table 2; Table 4; Table 6; Figure 6)
- `2510.09285#s2` — Spotlight on Token Perception for Multimodal Reinforcement Learning (https://arxiv.org/html/2510.09285v2; Sec 3.2.3 Eq. 5-6; Table 1; Table 2; Table 3; Table 5; Table 7; Theorem 3.1 (proof in Appendix D, not read in detail))
- `2603.25077#s1` — Bridging Perception and Reasoning: Token Reweighting for RLVR in Multimodal LLMs (https://arxiv.org/html/2603.25077v1; Sec 3; Table 3; Table 4; Appendix C Table 5)
- `2603.25077#s2` — Bridging Perception and Reasoning: Token Reweighting for RLVR in Multimodal LLMs (https://arxiv.org/html/2603.25077v1; Table 1; Table 2; Fig 2)
- `2606.00564#s2` — Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding (https://arxiv.org/html/2606.00564v1; Sec 4 Eq 16-18; Figure 4; Appendix B.3 Figure 8b; Appendix E Table 5)
