# Inject step-verifier scores into the RL return

Card `K22-process-reward-inside-rl` · status: candidate_pattern · members: 3 signatures from 3 full-text papers

## Operation

Add PRM or self-verifier step scores to the RL objective (per-step PPO rewards with value init from the PRM, a PRM drop gate on correct rollouts, verifier process rewards summed with the outcome).

- `2509.23738#s1`: query a separately served PRM for each step of a completed trajectory, add its positive/negative score plus a format reward as the per-step reward, compute GAE advantages, and update the policy with PPO-clip while initialising the value model from the PRM weights
- `2501.04686#s3`: for outcome-correct rollouts, compute the maximum relative drop between consecutive PRM step rewards and, if it exceeds rho, reduce the reward from 1 to 1-gamma; incorrect rollouts get 0; then compute group-relative advantages
- `2511.19900#s1`: after each rollout, switch the shared policy into Verifier mode to emit per-step (score, confidence, critique) with optional tool re-execution, convert these into process rewards with a tool-correctness weight and a cross-role regularizer, sum them (discounted) with the outcome reward, and update the shared policy with GRPO

**Transfer notes:** n/a (in-domain).

## When it works

- Per-step PRM rewards in PPO: AndroidWorld 32.8 -> 40.5 vs 37.0 with ORM [2509.23738#s1]
- Reducing the reward of correct rollouts with a large PRM drop: 6.8% vs 3.1% relative gain [2501.04686#s3]
- Self-verifier process reward over iterations: 57.3 -> 65.5 [2511.19900#s1]

## When it fails

- gamma=1 vanishes group rewards; too large/small rho reverts to vanilla GRPO [2501.04686#s3]
- No test of PRM hacking or drift [2509.23738#s1]
- Relies on an external-reward warm-up and outcome term; no hacking analysis [2511.19900#s1]

## Counterexamples

- Scalar PRM rewards (sum or per step) score below vanilla GRPO and shorten responses [2501.04686#s3]

## Sources

- `2509.23738#s1` — GUI-Shepherd: Reliable Process Reward and Verification for Long-Sequence GUI Tasks (https://arxiv.org/html/2509.23738v1; Table 1; Sec 4.1, 5.1; Appendix B.2)
- `2501.04686#s3` — Unlocking Multimodal Mathematical Reasoning via Process Reward Model (https://arxiv.org/html/2501.04686v6; Sec 4 Eq 5-6, Figures 4-6; Table 1; Table 4)
- `2511.19900#s1` — Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning (https://arxiv.org/html/2511.19900v2; Sec 3.2-3.3; Algorithm 1; Tables 1-3)
