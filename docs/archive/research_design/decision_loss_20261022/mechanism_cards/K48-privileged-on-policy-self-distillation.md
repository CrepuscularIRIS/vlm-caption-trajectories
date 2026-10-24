# Distil the same model under a richer condition into itself on its own rollouts

Card `K48-privileged-on-policy-self-distillation` · status: candidate_pattern · members: 8 signatures from 4 full-text papers

## Operation

Let the same model act as teacher under a privileged or richer condition (PI context, gold trace, clean image vs augmented view, terminal loop) and train the plain-condition student on its own rollouts toward the teacher's per-token distribution.

- `2602.04942#s1`: Sample trajectories from the PI-conditioned teacher; update the teacher with RL reward minus beta*KL(teacher || stop-grad student) and update the student with importance-weighted (pi_S/sg(pi_T)) reward plus a KL(sg teacher || student) term; mix the two objectives with weight alpha
- `2602.04942#s2`: Sample from the student and maximize reward minus beta*reverse KL(student || stop-grad PI-conditioned teacher), using the KL as a dense per-token signal (On-Policy Self-Distillation)
- `2606.10385#s1`: form log q_lambda = log q_part + lambda * (log q_full - log q_part) - C_t (renormalized, residual clipped at 5, small reference mixup) and minimize forward KL(q_lambda || pi_theta) on student rollouts
- `2608.14144#s1`: sample n=8 student rollouts on T(x) (downscale by s~U(0.3,0.6) without resizing back, then DDPM-step-200 Gaussian noise with prob 0.5), rescore each prefix with the EMA teacher on the clean image, and minimize generalized JSD (alpha=0.5) between teacher and student next-token distributions restricted and renormalized to the teacher's top-k tokens
- `2608.14144#s2`: train full runs that differ only in T across four families (information reduction, geometric, photometric, occlusion) and three strengths each, and relate final accuracy to the induced early teacher-student divergence
- `2608.14144#s3`: ablate view augmentation, teacher updating and divergence while keeping everything else fixed
- `2610.10623#s1`: sample responses from the intermediate-loop policy and minimize per-token reverse KL from the intermediate-loop distribution to the terminal-loop distribution, with the teacher either a frozen initial copy (LoopOPD) or the current stop-gradient terminal loop refreshed every step (D-LoopOPD)
- `2610.10623#s2`: bound the change of task surrogates at both loop depths after one step via gradient alignment between the distillation gradient and each depth's task gradient

**Transfer notes:** Brief mapping 2 inside one model. The VLM instance treats the clean image as the privileged view. Transfer to GT box/OCR as teacher-side context needs moderate teacher-student KL and positive PI utility (K05) and a check for answer shortcuts (K51). Most members are text-only LLMs.

## When it works

- Shared-parameter PI teacher/student: TravelPlanner 40.7-44.1% vs 27.5% RL [2602.04942#s1]
- Reverse KL to the PI-conditioned self as a dense signal beats RL [2602.04942#s2]
- Anchored residual between partial and full privileged views: 70.3 vs 67.9 [2606.10385#s1]
- Clean-image EMA teacher vs augmented-view student, no labels: perception avg 70.7 -> 77.4 [2608.14144#s1]
- Information-reducing views work best; the gain rises with moderate early divergence [2608.14144#s2]
- JSD beats forward and reverse KL; frozen teacher keeps 93% of the gain [2608.14144#s3]
- Terminal loop teaching an intermediate loop: AIME avg 55.5 -> 62.4 [2610.10623#s1]
- One-step joint improvement when distillation and task gradients align [2610.10623#s2]

## When it fails

- High initial teacher-student KL hurts; teacher ignores PI and collapses; negative PI utility fails [2602.04942#s1]
- Degrades OOD at 4B [2602.04942#s2]
- Single model family; unclear lambda [2606.10385#s1]
- Strong cropping removes evidence and lowers accuracy despite a large gap [2608.14144#s2]
- Without augmentation there is no gain [2608.14144#s3]
- Student depth must be chosen; local theory only [2610.10623#s1, 2610.10623#s2]

## Counterexamples

- Symmetric self-distillation (identity view) scores 65.2 vs base 70.6 [2608.14144#s2]

## Sources

- `2602.04942#s1` — Privileged Information Distillation for Language Models (https://arxiv.org/html/2602.04942v3; Sec 3.1; Table 1; Fig 4-7, Fig 9; Sec 7.1, Sec 8)
- `2602.04942#s2` — Privileged Information Distillation for Language Models (https://arxiv.org/html/2602.04942v3; Sec 3.2; Table 1; Fig 4; Fig 8; Sec 7.2)
- `2606.10385#s1` — Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation (https://arxiv.org/html/2606.10385v1; Sec 5 Eq 5-10; Table 1; Table 2; Appendix B Tables A.1-A.2)
- `2608.14144#s1` — Self-Supervised Visual On-Policy Distillation (https://arxiv.org/html/2608.14144v1; Sec 3.1 Eq 1-2; Sec 3.2 Eq 6; Tables 2-3)
- `2608.14144#s2` — Self-Supervised Visual On-Policy Distillation (https://arxiv.org/html/2608.14144v1; Sec 4.3; Figure 2(a)(b); Table 1)
- `2608.14144#s3` — Self-Supervised Visual On-Policy Distillation (https://arxiv.org/html/2608.14144v1; Sec 4.4; Tables 4-6)
- `2610.10623#s1` — Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models (https://arxiv.org/html/2610.10623v1; Sec 4.1-4.2; Tables 2-3; Fig. 3-5; App. C Tables 4-6)
- `2610.10623#s2` — Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models (https://arxiv.org/html/2610.10623v1; Sec 4.3; Appendix A)
