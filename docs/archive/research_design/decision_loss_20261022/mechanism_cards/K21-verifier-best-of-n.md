# Select among sampled candidates with a trained step verifier

Card `K21-verifier-best-of-n` · status: candidate_pattern · members: 5 signatures from 5 full-text papers

## Operation

Sample N candidate actions or solutions and pick by a process verifier or a self-critic tournament, without updating the policy.

- `2505.15277#s2`: at each step sample 5 candidate actions, score each with the PRM and execute the highest-reward one (Best-of-N reward-guided trajectory search); optionally, when the current step's reward is lower than the previous step's, have the policy refine its action using the PRM's textual feedback
- `2509.23738#s2`: sample n candidate actions per step, score each with the PRM, execute the highest-scoring one
- `2511.19900#s3`: use the Verifier as a process reward model to select the best of 8 sampled trajectories
- `2603.16253#s3`: score each candidate by geometric mean of gated step rewards (mapped to positive range) and pick the highest
- `2509.00676#s2`: Pair all candidates, have the model pick the preferred one in each pair (random choice on ties), and recurse in a knockout tournament until one response remains

**Transfer notes:** n/a (in-domain).

## When it works

- Web: 23.64 -> 34.55 success with an 8B PRM [2505.15277#s2]
- GUI: AndroidWorld 32.8 -> 37.9 [2509.23738#s2]
- Self-evolved verifier: +7.3% across policies [2511.19900#s3]
- Premise-gated PRM: +8.8 to +9.8 over Pass@1 [2603.16253#s3]
- Knockout tournament with the model's own pairwise preference: MathVision 30.6 -> 44.1 [2509.00676#s2]

## When it fails

- Saturates around n=8 [2509.23738#s2]
- Unreliable candidates break it (WorkArena Menu 25 -> 10) [2505.15277#s2]
- Worse than VisualPRM on MathVision/MathVerse-VO [2603.16253#s3]
- Large gap to oracle Best-of-128 [2509.00676#s2]
- No comparison with other trained PRMs [2511.19900#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2505.15277#s2` — Web-Shepherd: Advancing PRMs for Reinforcing Web Agents (https://arxiv.org/html/2505.15277v2; Sec 6.2, Tables 2-4, Sec 7.3, Appendix E.4 Fig 14)
- `2509.23738#s2` — GUI-Shepherd: Reliable Process Reward and Verification for Long-Sequence GUI Tasks (https://arxiv.org/html/2509.23738v1; Tables 1, 2, 5; Sec 4.2, 5.2)
- `2511.19900#s3` — Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning (https://arxiv.org/html/2511.19900v2; Sec 4.4; Table 4; Fig 3)
- `2603.16253#s3` — Grounding the Score: Explicit Visual Premise Verification for Reliable Vision-Language Process Reward Models (https://arxiv.org/html/2603.16253v2; Table 2; Sec 4.3; Appendix E, H)
- `2509.00676#s2` — LLaVA-Critic-R1: Your Critic Model is Secretly a Strong Policy Model (https://arxiv.org/html/2509.00676v1; Sec 3.2; Table 4; Figure 2)
