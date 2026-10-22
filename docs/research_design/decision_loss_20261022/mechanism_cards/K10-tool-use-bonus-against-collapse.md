# Add a bonus for tool calls so outcome-only RL does not abandon them

Card `K10-tool-use-bonus-against-collapse` · status: candidate_pattern · members: 6 signatures from 5 full-text papers

## Operation

Add a reward bonus for zoom/crop/frame tool use (usually only on correct answers, or when the query's tool-use rate is low, or per distinct coordinate) to keep the policy calling tools under outcome rewards.

- `2505.14362#s1`: Reward = accuracy + format + a tool bonus granted only when the answer is correct and >=1 zoom-in was made; optimize with GRPO over whole trajectories, masking loss on tool-returned observation tokens; KL coefficient 0, up to 6 tool calls
- `2509.24304#s1`: give an action bonus (lambda_gfn for get-frame-number >> lambda_cf for choose-frames) only when the trajectory ends in a correct answer, add it to the accuracy reward, omit the format reward, and optimise with GRPO without KL
- `2602.01334#s3`: Add a +0.1 bonus to responses that both call the tool and answer correctly, compared against outcome-only tool-RL and no-tool RL
- `2505.15966#s1`: add a curiosity bonus to responses that use pixel-space operations when the query's RaPR is below threshold H (clipped to zero above it), and subtract a penalty for operations beyond a maximum N, derived as a clipped Lagrangian relaxation of RaPR>=H and n_ops<=N constraints with fixed multipliers
- `2505.23678#s3`: Convert MCTS chains to dialog traces with crop tool calls and SFT, then run GRPO over full dialogs with observation tokens masked, a strict tag-automaton grammar reward, and a +0.2 bonus per coordinate >10px from previous ones (up to 4)
- `2509.24304#s3`: remove the format reward (malformed actions already terminate the trajectory) and do not reward number of turns k*(T-1)

**Transfer notes:** n/a (in-domain). Read together with K46: tool-free evaluation shows how much of the gain the tool itself carries.

## When it works

- Conditional (correct and >=1 call) bonus: HR-4K 75.1 vs 53.4 without tool reward; without it tool calls stop [2505.14362#s1]
- Correctness-conditioned action bonus avoids the collapse seen with unconditional bonuses [2509.24304#s1]
- Curiosity bonus when query-level tool rate is low restores tool use from 0 to about 80% [2505.15966#s1]
- Per-coordinate diversity bonus prevents collapse to single-turn outputs [2505.23678#s3]

## When it fails

- Unconditional bonus causes repeated identical actions; turn-count reward collapses with degenerate reasoning; a format reward makes action use drop [2509.24304#s1, 2509.24304#s3]
- Without self-correction warm-start the bonus is hacked: superficial tool calls, answers by text reasoning [2505.15966#s1]
- Diversity bonus rewards distinct coordinates regardless of usefulness [2505.23678#s3]
- Text-only CoT RL on the same data reaches 88.5 on V* vs 90.1, so the tool margin is mostly on HR-8K [2505.14362#s1]

## Counterexamples

- A +0.1 call bonus raises call intensity and Call Harm but not sustained correction of hard tool-free failures [2602.01334#s3]

## Sources

- `2505.14362#s1` — DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning (https://arxiv.org/html/2505.14362v3; Eq 2; Sec 3.2; Table 5; Fig 4; Table 1)
- `2509.24304#s1` — FrameThinker: Learning to Think with Long Videos via Multi-Turn Frame Spotlighting (https://arxiv.org/html/2509.24304v2; Sec 3.3.2-3.3.3; Fig 4(d); Tables 2-3; Appendix D.1)
- `2602.01334#s3` — What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom (https://arxiv.org/html/2602.01334v2; Appendix C; Fig 6-7)
- `2505.15966#s1` — Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning (https://arxiv.org/html/2505.15966v3; Sec 4; Sec 5.2; Table 1; Fig 5-7; Appendix B, C.3)
- `2505.23678#s3` — Grounded Reinforcement Learning for Visual Reasoning (https://arxiv.org/html/2505.23678v3; Sec 4.3.1; Tables 1, 3; Fig A1)
- `2509.24304#s3` — FrameThinker: Learning to Think with Long Videos via Multi-Turn Frame Spotlighting (https://arxiv.org/html/2509.24304v2; Sec 3.3.1, 3.3.4; Fig 4(b); Appendix D.2)
