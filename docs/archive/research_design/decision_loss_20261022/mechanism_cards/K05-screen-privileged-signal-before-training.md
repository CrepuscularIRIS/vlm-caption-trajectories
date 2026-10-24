# Screen candidate privileged information with a cheap statistic before training

Card `K05-screen-privileged-signal-before-training` · status: candidate_pattern · members: 4 signatures from 3 full-text papers

## Operation

Before committing to a privileged signal or teacher context, measure its informativeness or distance (residual dependence with return, value-prediction gain, initial teacher-student KL, PI utility) and choose by that measurement.

- `2509.26000#s2`: regress return G_t and signal i_t on (encoded history, action) with cross-fitted random forests, then test residual dependence by HSIC with episode-level permutation; declare the signal alpha-residual-informative if p<alpha
- `2509.26000#s3`: compute per-episode reduction in squared return-prediction error from adding i_t and run a one-sided bootstrap or t-test that the mean gain exceeds epsilon
- `2602.04942#s3`: Measure initial teacher-student KL and utility of each PI type to predict which PI format and alpha will transfer
- `2609.25623#s2`: measure mean teacher-to-student KL induced by each context at initialization (and in windows around the final boxed answer) and compare its ordering with post-training utility

**Transfer notes:** For VLMs: screen candidate privileged contexts (GT box, crop, OCR string, caption, answer) by base-model teacher-student KL and PI utility before self-distillation or asymmetric rewards. The KL evidence conflicts across papers (text-only math), so utility must be measured too.

## When it works

- HSIC residual test on random-policy episodes picks the reward-relevant state subsets [2509.26000#s2]
- Per-episode reduction in return-prediction error from adding the signal ranks the same subsets [2509.26000#s3]
- Higher initial teacher-student KL correlated with lower final performance for pi-Distill; information content predicted OPSD [2602.04942#s3]

## When it fails

- Residual independence is necessary, not sufficient; validated only on 20-state synthetic POMDPs [2509.26000#s2]
- Value-prediction utility is not policy improvement [2509.26000#s3]
- Observational analysis without control of all variables [2602.04942#s3]

## Counterexamples

- Full solution has the largest initial KL but not the best utility at 4B/8B: KL does not monotonically order privileged contexts [2609.25623#s2]

## Sources

- `2509.26000#s2` — Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access (https://arxiv.org/html/2509.26000v3; Sec 4.1 Algorithm 1; Sec 5.2 Table 1; Appendix G.2-G.4)
- `2509.26000#s3` — Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access (https://arxiv.org/html/2509.26000v3; Sec 4.2; Table 1; Appendix G.3)
- `2602.04942#s3` — Privileged Information Distillation for Language Models (https://arxiv.org/html/2602.04942v3; Sec 7; Fig 5, Fig 6, Fig 8)
- `2609.25623#s2` — What Should a Self-Teacher See? Privileged Context Design for On-Policy Self-Distillation (https://arxiv.org/html/2609.25623v2; Sec 3.4 Eq 6; Sec 4.3; Figure 2)
