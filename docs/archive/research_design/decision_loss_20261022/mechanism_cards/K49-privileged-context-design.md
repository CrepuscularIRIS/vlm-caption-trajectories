# Choose what the self-teacher sees

Card `K49-privileged-context-design` · status: candidate_pattern · members: 5 signatures from 4 full-text papers

## Operation

Decide or adapt the self-teacher's privileged context (compiled abstract hints, answer only, retrieved skill/mistake bank, evidence anchors, an adaptively truncated reference).

- `2609.25623#s1`: compile the reference solution coarse-to-fine into an abstract hint (category L4, then framing L3 conditioned on L4, then strategy L2 conditioned on L3-L4), and use that hint instead of the full solution as the teacher's privileged context when scoring student rollouts with a clipped forward-KL objective
- `2609.25623#s3`: replace the full reference solution with the bare answer (or a short prefix plus answer) as the teacher's privileged context in OPSD
- `2605.28791#s2`: summarize successful traces into general-skill entries and failed traces into mistake entries, merge redundant candidates, write back as dynamic entries every 25 steps (skipped when recent success is high), prune to a capacity limit, and pair rank-aligned skill and mistake entries per teacher
- `2608.12764#s1`: prompt a strong LLM with question and gold answer to extract as many step-level evidence snippets (source title + explanation) as possible, drop QA pairs whose anchors have inaccessible URLs, and insert the anchors plus the student's wrong answer into the teacher prompt with an instruction not to answer directly from anchors
- `2608.08176#s1`: each batch, weight tokens by w_t = sigmoid((g_t - lambda d_t)/tau), train on sum_t w_t g_t / sum_t w_t, compute load = mean w_t d_t, raise lambda by eta_lambda (load - eps) (projected at 0), and move beta by eta_beta (eps - load) clipped to [beta_min, 1]

**Transfer notes:** All members are text-only. For VLMs the candidate contexts are GT box, crop, OCR string, caption and answer; the text evidence says more context is not better.

## When it works

- Abstract hints beat the full solution at 4B/8B by 1.4-1.6 [2609.25623#s1]
- Answer-only context is within 0.2 of the full solution at 4B/8B [2609.25623#s3]
- Skill/mistake bank helps only as teacher-side context [2605.28791#s2]
- Evidence anchors from a search LLM give the best single teacher context [2608.12764#s1]
- Adapting how much reference the teacher sees by a load budget: 58.7 vs 56.4 [2608.08176#s1]

## When it fails

- At 1.7B the full solution wins; winners vary by compiler and task; leakage up to 4% [2609.25623#s1]
- Answer-only trails at 1.7B [2609.25623#s3]
- Skills in the student prompt can underperform the base [2605.28791#s2]
- Anchors collapse the teacher to few tool calls [2608.12764#s1]
- Sensitive to the budget epsilon [2608.08176#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2609.25623#s1` — What Should a Self-Teacher See? Privileged Context Design for On-Policy Self-Distillation (https://arxiv.org/html/2609.25623v2; Sec 3.3 Eq 5; Sec 4.1 Table 1; Sec 4.2 Table 3; Appendix A.1)
- `2609.25623#s3` — What Should a Self-Teacher See? Privileged Context Design for On-Policy Self-Distillation (https://arxiv.org/html/2609.25623v2; Sec 4.1 Answer-only Context; Sec 4.2 Semantic Rewriting vs Prefix Truncation; Table 1)
- `2605.28791#s2` — Skill-Conditioned Gated Self-Distillation for LLM Reasoning (https://arxiv.org/html/2605.28791v4; Sec 2.1, 2.6; Table 3; Fig 4; Table 12)
- `2608.12764#s1` — Beyond Outcome Rewards: Step-Level Self-Distilled Policy Optimization for Deep Search Agents (https://arxiv.org/html/2608.12764v1; Sec 3.1; Table 1; Table 5; Appendix G)
- `2608.08176#s1` — Matching Supervision to the Student's Learning Capacity: A Unified Framework for On-Policy Self-Distillation (https://arxiv.org/html/2608.08176v1; Sec 3.4 Eq 7-10, Algorithm 1; Tables 1-3; Fig 5; Proposition 1 (theorem, conditional on Assumption 1))
