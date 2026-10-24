# Pre-registration: first-action decision loss MVE (2026-10-23)

Written before any GPU use for this test. It will be entered into study `rl-objective-credit` as a collection-only `plan_test` after the four-case (023/024) and zero-arm interprets. The text below is copied verbatim.

Design source: mentor's 10-22 suggestion (private conversation log, lines 170–180; summarised in [docs/research_design/decision_loss_20261022/](../../../docs/archive/research_design/decision_loss_20261022/README.md)) [scrub: private link removed], and Pro's design `docs/research_design/decision_loss_20261022/PRO_RESPONSE.md` (UTF-8 sha256 55df6e12…), §§2–8. The user asked for it to be carried out as designed.

## Question

Starting from the same weights, under the same protocol, number of main rollouts and number of updates: does adding "forked branch values + first-action label supervision" make the first action follow independent branch-value differences better, without clearly hurting accuracy or recovery? What does it cost?

This is not compute-matched, and it does not claim that BCE beats other losses that use the same Q information.

## Arms (identical except β)

| | main | control |
|---|---|---|
| Protocol | v5.2i (Step 1's WORD is intent only; `caption_v52i.py`) | same |
| Start | r1 (RLOO λ=.01) step-40 HF weights, plus r1's `dataloader.pt` (continuing questions 41–60 in r1's order) | same |
| Optimizer | Fresh AdamW. lr 1e-6, constant, **no warmup** (revision 1), grad-norm 0.1. Global batch = 12 questions × 16, one optimizer step per update (on-policy shortcut) | same |
| RLOO | r = Y(1−λN), λ=.01, leave-one-out over 16; KL off; no dynamic sampling | same |
| Decision loss | β = 1/256, K = 4 + 4 forks per question at one pre-assigned slot's first post-caption prefix h; H0 = 12 | β = 0, no forks |
| Updates | 20 | 20 |

Labels: the full tag log-prob sums are taken for `<answer` (2 tokens) and `<grounding` (3 tokens). The policy merges `>` with the following text, so `>` is not part of the label (Qwen3-VL tokenizer check, 10-22). h must end with `</think>` `\n`. Any other boundary makes the state unusable: it contributes 0 and stays in H0. The loss is l = relu(D)·softplus(−z) + relu(−D)·softplus(z), with D = mean R_answer − mean R_crop computed per fork and z = s_A − s_C at T = .7. The aux term is β/H0·Σ l, added before the single optimizer step; nothing is renormalised by Σ|D| or by the token count.

## Evaluation set

64 questions: E1 (552) minus the 20 reviewed uids, then a seeded shuffle (seed 20261023), first 64. The list is `dl_questions.json` (sha256 aaee25c1…). The train-image overlap check by path found 0. E1 has been used repeatedly, so **every result here is a development diagnostic, not a confirmation**. The first 16 questions in list order are the recovery questions.

## Measurements

1. **Layer 1 (diagnostic only):** aux loss, p, tag mass m, usable-state share, fork terminals, sign counts of D.
2. **Layer 2:** reference = the common start (r1 step 40). It writes turn 1 once per question. At each usable h, 4 + 4 forks are drawn with an RNG namespace disjoint from training, giving δ_audit(h). U_choice = (1/64)·Σ_h (p_main(h) − p_ctrl(h))·δ_audit(h); unusable h contribute 0. The 95% interval is a percentile bootstrap over questions (10,000 draws, seed 0).
3. **Layer 3:** both final models, clean and E2 (`eval_v4 --perturb first_crop`), 2 samples per question. Measured: strict accuracy (`ok_train_rule`), R = Y(1−.01N), crops, generated tokens, invalid share, share of direct first answers. Main − control differences use a paired bootstrap over questions.
4. **Recovery:** for the 16 recovery questions, the reference's first turn is used only if it crops the original image. That crop is shifted by the E2 rule, executed, and continued twice by each final model. Missing states are reported, not replaced.
5. **Cost:** seconds per step for each arm, fork generated tokens, GPU-hours.

## Decision rule (Pro §8.5; screening margin 2 points, not a licence for a permanent loss)

| Verdict | Rule |
|---|---|
| **Supported** | U_choice CI lower bound > 0, **and** clean strict-accuracy difference (main − control) point estimate ≥ −0.02 with CI upper bound ≥ 0, **and** no clear harm (CI upper bound < 0) on E2 accuracy or recovery accuracy, **and** main's invalid share no more than 0.05 above control |
| **Refuted** | U_choice CI upper bound < 0, **or** clear harm (CI upper bound < 0) on clean, E2 or recovery accuracy, **or** main's invalid share more than 0.05 above control |
| **Undecided** | Anything else. The budget stops the run; this is neither a refutation of the mechanism nor evidence that accuracy did not drop |

None of these counts as success on its own: a lower CE, a higher direct-answer share, or fewer crops.

## Gates and budget

- **Before GPU:** `test_decision_loss.py` must pass, including the real-tokenizer test with the training tokenizer.
- **Step 1 smoke (main):** `[dl_forks] usable > 0`; `[dl_aux]` present on both ranks; exactly one optimizer step. A failure means no further updates.
- **Hard cap:** 8 GPU-hours in total (training for both arms plus all evaluation), 4M generated tokens, 50M processed tokens; whichever comes first. After step 1, if (t_main + t_ctrl) × 20 × 2 GPUs + 1.5 GPU-hours of evaluation is projected above 8 GPU-hours, both arms stop and the run is reported as "budget not met". K, the crop cap and the branch budgets are not reduced to make it fit.
- **One run, final.** No re-runs with other β, K, λ or masks.

## Revisions (all before any GPU use)

1. **Warmup 5% → 0.** EasyR1 `get_constant_schedule_with_warmup` gives int(0.05×20) = 1 warmup step with lr(step 0) = 0/1 = 0, so update 1 would be empty; r1 at its step 41 runs at constant 1e-6. Both arms changed alike. Source: independent code review 10-23.
2. **Extra step-1 gate:** both logs must contain `[caption] dataloader state loaded` (otherwise the question order did not continue r1's 41–60; stop).
3. **Engineering fixes (design unchanged):** fork log one file per process with one write per call; aux log-prob sum via masked_fill; a truncated reference turn 1 is not used as a recovery state.

## Deviations during execution (recorded after GPU use; design, samples, metrics and decision rule unchanged)

4. **Step-1 gate applied by hand at step 8.** The local runner's gate query ended in `[ -f DONE ] && echo …`, which exits 1 while DONE is absent, so every poll counted as an ssh failure and the gate never ran. At 07:51 CST (main 8, ctrl 10 updates) the same checks were applied by hand: usable 6/6, `[dl_aux]` 16 lines, dataloader loaded in both logs, no Traceback; projection (194.6 + 178.7) × 20 × 2 / 3600 + 1.5 = 5.65 GPU-h ≤ 8 → continue. Runner replaced by `collect_dl.sh` (wait + copy only).
5. **Post-training stages re-run twice.** Audit/eval first failed on import (`CAPTION_GRPO_CODE` unset), then on the training env contract (`CAPTION_STD_FLOOR`, `CAPTION_ROLLOUT_LOG`, `CAPTION_DL_H0`, `CAPTION_DL_LOG`). `dl_remote.sh` now exports these for post-training stages only (β = 0, logs to `evidence/…/post_*.jsonl`). Training and merged weights were not touched; the failed logs are kept in `failed_post1/`, `failed_post2/`.
