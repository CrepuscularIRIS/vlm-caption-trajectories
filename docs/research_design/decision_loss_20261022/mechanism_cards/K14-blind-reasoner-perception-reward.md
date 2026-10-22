# Reward perception text by whether a vision-free reasoner can answer from it

Card `K14-blind-reasoner-perception-reward` · status: candidate_pattern · members: 5 signatures from 3 full-text papers

## Operation

Score the policy's externalized perception text (caption, recognition block) by whether a frozen vision-free LLM reaches the gold answer from it, and use that as reward or to split credit between perception and reasoning tokens.

- `2509.22647#s1`: For each sampled caption, have a vision-free LLM answer N randomly sampled image-specific MCQs from the caption alone with shuffled options; set the caption reward to the mean exact-match accuracy and compute GRPO advantages with a KL penalty
- `2605.14054#s1`: feed the question plus the rollout's perception blocks to a blind text-only LLM and give a binary perception reward if it reaches the gold answer; penalize perception blocks longer than a token limit (~800)
- `2605.14054#s2`: for wrong-answer rollouts, add alpha_protect*|A| to perception-token advantages when R_P=1 ('bad thinking') and subtract alpha_punish*|A| when R_P=0 ('bad seeing'), with trajectory return R_O + lambda*R_P
- `2509.26594#s1`: let the frozen reasoner optionally ask one clarification question about a caption, answer it with the frozen reference captioner, and reward the caption 1 (correct without clarification), alpha (correct with clarification) or 0, updating only the initial-caption tokens with BNPO and a KL penalty
- `2509.26594#s3`: prove REINFORCE with an action-independent baseline stays unbiased under 0/alpha/1 reward; if the reasoner shares parameters, add its score term or stop gradients

**Transfer notes:** n/a (in-domain). Requires visual evidence that can be stated as discrete text premises.

## When it works

- Caption reward = accuracy of a blind LLM on shuffled image-specific MCQs: CapRL-3B comparable to Qwen2.5-VL-72B in Prism [2509.22647#s1]
- Binary perception-sufficiency reward: removing it costs 6.9 on V*; 86% agreement with human sufficiency majority [2605.14054#s1]
- Protect or punish perception-token advantages in failed rollouts by that verdict: +3 on perception tasks over naive shaping [2605.14054#s2]
- Tiered reward when the reasoner needs a clarification: +3.1 avg over binary RL [2509.26594#s1]
- Unbiased if the reasoner and clarifier do not share the policy's parameters [2509.26594#s3]

## When it fails

- With one MCQ the LLM's option bias misdirects the reward; leaking QA lowers results; only probed information is measured [2509.22647#s1]
- Non-verbalizable visuals are out of scope; the blind reasoner can succeed from its prior (4.5% false positives) [2605.14054#s1]
- WeMath regresses; math benchmarks only [2509.26594#s1]
- Shared-trunk reasoners are not covered by the unbiasedness argument [2509.26594#s3]
- Single-run ablations [2605.14054#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2509.22647#s1` — CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning (https://arxiv.org/html/2509.22647v1; Sec 3.1; Sec 4.2; Table 3; Fig 1)
- `2605.14054#s1` — Bad Seeing or Bad Thinking? Rewarding Perception for Multimodal Reasoning (https://arxiv.org/html/2605.14054v2; Sec 2.3; Table 2; Table 3; Appendix E.2)
- `2605.14054#s2` — Bad Seeing or Bad Thinking? Rewarding Perception for Multimodal Reasoning (https://arxiv.org/html/2605.14054v2; Sec 2.5; Table 1; Table 2; Appendix A)
- `2509.26594#s1` — Clarification as Supervision: Reinforcement Learning for Vision-Language Interfaces (https://arxiv.org/html/2509.26594v1; Tables 1-2; Sec 3.3, 4.1-4.2; Appendix A, C)
- `2509.26594#s3` — Clarification as Supervision: Reinforcement Learning for Vision-Language Interfaces (https://arxiv.org/html/2509.26594v1; Appendix B)
