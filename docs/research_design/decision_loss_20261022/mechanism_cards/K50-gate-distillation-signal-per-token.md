# Do not apply teacher-student gaps uniformly: gate them

Card `K50-gate-distillation-signal-per-token` · status: candidate_pattern · members: 3 signatures from 3 full-text papers

## Operation

Filter the per-token or per-step teacher-student gap before using it (calibrate against context-induced shift intervals, set polarity by outcome sign and support, apply step weights only on incorrect trajectories).

- `2609.21619#s2`: per token, set an interval [l_T - lambda*d_down, l_T + lambda*d_up] from the maximal down/up teacher shifts across the two interventions, replace the OPD advantage with the student log-prob's distance outside this interval (zero if inside), and train with the standard OPD objective
- `2605.28791#s1`: for each teacher compute per-token gaps Delta_t = log p_teacher(y_t) - log p_student(y_t) on the sampled rollout, average the masked and clipped gaps into a support score, set polarity = sign(r) * sign(support) (0 if |support| <= threshold), and minimize sum_k alpha_k * polarity_k * mean_t gate(Delta_t), where gate(D) = log2 - log(1 + exp(-D^2 / 2 tau)) and alpha_k is the softmax of retrieval scores
- `2608.12764#s2`: teacher-force each student step under teacher and student contexts, take the stop-gradient log joint-likelihood ratio Delta per step, set a shared step weight w = min(exp(sign(A) * Delta), 1+eps) on all tokens of that step, multiply the advantage by w only for trajectories with incorrect final answers, leaving correct trajectories with plain GRPO

**Transfer notes:** Text-only members. The VLM analogue is a visual-dependency gate (K17, 2606.00564#s2).

## When it works

- Shift-interval calibration: 50.8 -> 53.1 and shorter responses [2609.21619#s2]
- Outcome-signed polarity gating: 43.7 vs 42.0 OPSD at 1.7B; no polarity collapses late [2605.28791#s1]
- Step weights only on incorrect trajectories beat GRPO and token-level weighting [2608.12764#s2]

## When it fails

- Solution-level interventions over-filter; large lambda degrades [2609.21619#s2]
- Below OPSD at 8B; reference answer collapses training [2605.28791#s1]
- Cannot change update direction; no process signal on correct trajectories [2608.12764#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2609.21619#s2` — Calibrating Teacher--Student Discrepancy for On-Policy Distillation (https://arxiv.org/html/2609.21619v1; Sec 3 Eq 13-14; Table 3; Table 4; Fig 4-5; Appendix A.9 Table 13)
- `2605.28791#s1` — Skill-Conditioned Gated Self-Distillation for LLM Reasoning (https://arxiv.org/html/2605.28791v4; Sec 2.3-2.5; Table 3; Table 4; Fig 3-4; Appendix F)
- `2608.12764#s2` — Beyond Outcome Rewards: Step-Level Self-Distilled Policy Optimization for Deep Search Agents (https://arxiv.org/html/2608.12764v1; Sec 3.2 Eq 6-8; Table 3; Table 4; Fig 4-5; Appendix E Table 9)
