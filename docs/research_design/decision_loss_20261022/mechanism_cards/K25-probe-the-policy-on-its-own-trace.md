# Re-query the policy on its own trace with probes whose answers are known

Card `K25-probe-the-policy-on-its-own-trace` · status: candidate_pattern · members: 3 signatures from 2 full-text papers

## Operation

Append an auxiliary probe with a known answer (a labelled visual claim, the same question with shuffled options) to the policy's own partial or complete reasoning trace, read its answer, and add probe correctness to the reward.

- `2509.25848#s1`: at K=20 random positions in each sampled reasoning trace, truncate the trace, append a visual claim and read the policy's argmax yes/no judgment; score against the claim label, aggregate with exponentially later-weighted weights w_k=exp(beta*a_k/T), and add gamma*1[answer correct]*R_perc to accuracy+format reward before group normalization
- `2509.25848#s3`: prompt the training policy to generate balanced true/false visual claims about the image during rollout, then use these self-generated claims as anchors for the perception reward
- `2510.10104#s1`: after each rollout, re-prompt the policy with the same media, the same reasoning trace and a shuffled option list to decode a second answer; add r_c=alpha1 if both answers agree and are correct, alpha2 if they disagree and exactly one is correct, alpha3 if they agree and are both wrong, else 0, to the accuracy/format/length reward before group normalization

**Transfer notes:** n/a (in-domain).

## When it works

- Claim probes at 20 truncation points, later-weighted and gated by answer correctness: +4.37 over GRPO [2509.25848#s1]
- Self-generated claims come close to GPT-5 claims [2509.25848#s3]
- Second answer under shuffled options must agree: video avg 0.615 -> 0.637 [2510.10104#s1]

## When it fails

- Non-visual or question-aligned claims give smaller gains; hyperparameters differ by task type [2509.25848#s1]
- Self-generated claim labels unaudited [2509.25848#s3]
- MCQ only; checks answer invariance, not visual use [2510.10104#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2509.25848#s1` — More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models (https://arxiv.org/html/2509.25848v3; Sec 4.2 Eq 3-5; Tables 1-3; Fig 5-7; Appendix B.4-B.5, B.9, B.13)
- `2509.25848#s3` — More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models (https://arxiv.org/html/2509.25848v3; Appendix B.12 Table 15; B.15 Table 18)
- `2510.10104#s1` — Answer-Consistent Chain-of-thought Reinforcement Learning For Multi-modal Large Langauge Models (https://arxiv.org/html/2510.10104v1; Sec 4 Eq 3-6; Tables 1-4; Sec 5.3)
