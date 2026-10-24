# Reward a VLM agent by the error of its predicted next observation

Card `K56-cross-modal-prediction-curiosity` · status: case · members: 3 signatures from 1 full-text papers

## Operation

Project the hidden state at the agent's predicted next-state text into visual feature space, use the error against an EMA vision encoder's embedding of the real next observation as intrinsic reward, update the encoder separately, and reset the projector when the loss plateaus.

- `2605.03782#s1`: project the prediction-token hidden state through a 2-layer MLP into the visual feature space, compute normalized MSE to the target encoder's embedding of o_{t+1}, and add beta times this error as an intrinsic reward to the extrinsic reward before Bi-level GAE/PPO
- `2605.03782#s2`: minimize the language-to-vision prediction loss with LLM weights frozen so that only projector and online vision encoder are updated, then separately run PPO on full parameters, and update the target encoder by EMA
- `2605.03782#s3`: when |L_t - L_{t-1}| stays below epsilon=0.1 for K consecutive steps (K=30 Sokoban/FrozenLake, 20 others), randomly re-initialize the projector while keeping the evolved vision encoder

**Transfer notes:** n/a (multi-turn visual agents).

## When it works

- Overall 0.86 vs 0.81 VAGEN-Full [2605.03782#s1]
- EMA target needed for stability [2605.03782#s2]
- Projector resets precede success increases [2605.03782#s3]

## When it fails

- Small gains on several tasks; noisy-TV risk not analysed [2605.03782#s1]
- Curve-only evidence [2605.03782#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2605.03782#s1` — What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (https://arxiv.org/html/2605.03782v1; Table 1; Table 2; Sec 3.3, 4.2-4.3; Figure 4)
- `2605.03782#s2` — What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (https://arxiv.org/html/2605.03782v1; Sec 3.2; Figure 3; Appendix A algorithm Steps A-C)
- `2605.03782#s3` — What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (https://arxiv.org/html/2605.03782v1; Sec 3.4; Sec 4.3; Figures 3, 6; Appendix B.7)
