# Measure where privileged teacher-student gaps arise and whether they carry shortcuts

Card `K51-privileged-teacher-discrepancy-diagnostics` · status: candidate_pattern · members: 3 signatures from 2 full-text papers

## Operation

Rescore fixed student rollouts under varied teacher contexts and measure where and on which tokens gaps arise, and count shortcut evidence in trained outputs.

- `2609.21619#s1`: insert contrasting contexts before the student rollout in the teacher prompt, measure per-token teacher log-prob shifts, and quantify where shifts occur, whether opposite-semantics contexts shift in the same direction, and which token forms shift most
- `2606.10385#s2`: measure top-k (k=16) disagreement and teacher mass on tokens the student assigns probability below tau, bucketed by rollout position and privileged-context length; score each rollout span by summed log-ratio advantages (full-student, partial-student, full-partial) and use them as correctness classifiers
- `2606.10385#s3`: count responses with shortcut-like evidence (references to an unavailable reference solution, answer-conditioned certainty without local derivation, prompt-external answer cues), once per response

**Transfer notes:** Would let a VLM study check whether GT-box/OCR context shifts visual tokens or only discourse tokens. Untested on images.

## When it works

- Discourse tokens shift >89% under any context; opposite-semantics contexts shift in the same direction [2609.21619#s1]
- Gaps grow near rollout tails and with privileged context length [2606.10385#s2]
- Shortcut events are rarer with anchored residual guidance [2606.10385#s3]

## When it fails

- Math long-CoT only [2609.21619#s1, 2606.10385#s2]
- Small absolute counts; detector rules unspecified [2606.10385#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2609.21619#s1` — Calibrating Teacher--Student Discrepancy for On-Policy Distillation (https://arxiv.org/html/2609.21619v1; Sec 2.2-2.3; Fig 2-3; Table 2; Appendix A.3-A.4 (not read in detail))
- `2606.10385#s2` — Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation (https://arxiv.org/html/2606.10385v1; Sec 4.1-4.2; Figures 3-4; Appendix C.2-C.3; Figure A.1)
- `2606.10385#s3` — Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation (https://arxiv.org/html/2606.10385v1; Sec 6.3; Figure 6; Appendix C.4)
