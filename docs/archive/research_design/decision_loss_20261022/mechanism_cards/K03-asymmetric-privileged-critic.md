# Train the critic on privileged state while the actor sees only observations

Card `K03-asymmetric-privileged-critic` · status: candidate_pattern · members: 6 signatures from 5 full-text papers

## Operation

Condition the value function on training-only privileged state or signals (plus the actor's history) and update an observation-only actor with its advantages; discard the critic at deployment.

- `2406.12505#s1`: train the critic on privileged simulator state while the actor is conditioned only on pixels, and discard the critic at deployment
- `2412.06313#s1`: store paired (clean state, corrupted observation) transitions; train critics on clean state s, train the actor on corrupted observation o by maximizing Q(s, pi(o)), and compute the obstacle-distance penalty from the privileged map rather than from the noisy depth image
- `2506.05516#s2`: train PPO with an asymmetric actor-critic: value estimates from privileged state, policy from noise-injected observations deployable on the real robot
- `2509.26000#s1`: train a recurrent critic V(h_t, i_t) by TD learning and use its TD-error advantage to update a history-only actor
- `2412.00985#s2`: update the finite-memory policy synchronously over all memories by a multiplicative-weights / NPG step on the belief-weighted expected privileged Q, E_{s~b^apx(z)}[Q(z,s,.)], with optimistic bonuses in policy evaluation
- `2412.00985#s3`: estimate an approximate POMDP model from privileged-state exploration, truncate rarely visited states by redirecting their probability mass to other states, and compute beliefs in the truncated model

**Transfer notes:** Brief mapping 2 source. In VLM RL, GRPO has no learned critic, so transfer needs a PPO value model or a baseline conditioned on GT boxes/OCR. Must hold: the privileged signal predicts return beyond the observation history (screen with K05). VLM reward-side instances are in K04.

## When it works

- Pixel-only (symmetric) critics fail to learn (0% success) while privileged critics reach 90-100% [2406.12505#s1]
- Clean-state critics with a corrupted-observation actor: 86.7% vs 33.3% success for plain TD3 [2412.06313#s1]
- Partial privileged signals suffice: the informed policy gradient is unbiased and matches full-state critics with less information [2509.26000#s1]
- Critic must condition on history and state; belief-weighted privileged Q gives polynomial sample complexity in gamma-observable POMDPs [2412.00985#s2, 2412.00985#s3]
- Used for zero-shot sim-to-real with noisy proprioception actors [2506.05516#s2]

## When it fails

- Uninformative or noisy extra features degrade value estimation; symmetric critics win on some tasks [2509.26000#s1]
- A state-only critic is biased; theory and experiments are tabular only [2412.00985#s2]
- Actor memory limits: success drops when the gate is not visible for several frames [2406.12505#s1]
- No ablation isolating the asymmetric critic [2506.05516#s2]
- Privileged critic and privileged reward contributions are not separated [2412.06313#s1]

## Counterexamples

- On Repeat First symmetric A2C performs best, and on Count Recall no signal beats full state [2509.26000#s1]

## Sources

- `2406.12505#s1` — Demonstrating Agile Flight from Pixels without State Estimation (https://arxiv.org/html/2406.12505v1; Sec III-A3; Fig 4; Table I; Appendix VI-A)
- `2412.06313#s1` — Vision-Based Deep Reinforcement Learning of UAV Autonomous Navigation Using Privileged Information (https://arxiv.org/html/2412.06313v1; Sec 3.2.1-3.2.2; Table 3; Fig 6; Fig 8 (Sec 4.3))
- `2506.05516#s2` — Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots (https://arxiv.org/html/2506.05516v2; Sec III-B; Fig 3)
- `2509.26000#s1` — Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access (https://arxiv.org/html/2509.26000v3; Sec 3.2 Theorem 3.1; Appendix A; Fig 1; Appendix G.1 Table 6)
- `2412.00985#s2` — Provable Partially Observable Reinforcement Learning with Privileged Information (https://arxiv.org/html/2412.00985v3; Sec 3.2 Eq 3.3; Sec 5.1 Theorem 5.1; Theorem 5.3; Sec 6 Figure 2)
- `2412.00985#s3` — Provable Partially Observable Reinforcement Learning with Privileged Information (https://arxiv.org/html/2412.00985v3; Sec 5.2; Theorem 5.2)
