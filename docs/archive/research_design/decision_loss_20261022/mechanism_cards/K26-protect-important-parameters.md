# Locate parameters carrying prior abilities and keep updates away from them

Card `K26-protect-important-parameters` · status: candidate_pattern · members: 6 signatures from 4 full-text papers

## Operation

Identify parameters or directions important for existing abilities (delta magnitude and Hessian sensitivity, earlier tasks' largest LoRA positions, high-activation neurons on calibration data, gradient subspace of anchor examples) and revert, regularize, freeze or project updates accordingly.

- `2402.12048#s1`: score each fine-tuned parameter by omega * normalized |theta_sft - theta_pre| + (1-omega) * normalized OBS sensitivity (theta_sft - theta_pre)^2 / (2[H^-1]_mm), keep the top-rho fraction as fine-tuned values and revert all others to pre-trained values
- `2402.12048#s2`: for each reverted parameter m, add the OBS compensation -(theta_sft^m - theta_pre^m)/[H^-1]_mm * H^-1[:,m] to the kept parameters so the layer output on target inputs stays close to the fine-tuned layer output
- `2505.02486#s2`: after finishing a task, record the positions of its largest |Delta W| elements as a mask and merge the LoRA; when training each later LoRA, add lambda * sum_i |Delta W_j| (Hadamard) R_i to the loss, pushing the new update toward zero only at those positions
- `2608.11758#s1`: for each target linear layer, take the L2 norm of each output neuron's activation over the sequence, L2-normalize across neurons within each sample, average over samples, globally rank neurons, and zero the gradients of the top-p (30%) rows via a binary mask while fine-tuning the remaining rows; keep MLP blocks and output projection frozen
- `2608.11758#s2`: before each new task in a continual sequence, recompute neuron saliency on MMMU plus samples from earlier tasks and rebuild the freeze mask
- `2601.18255#s1`: before training task k, run a few fine-tuning steps on the anchor set on a temporary copy, stack the per-step LoRA gradients, take the top-r left singular vectors as a historical subspace U_hist, then train task k from the original parameters with every gradient replaced by (I - U_hist U_hist^T) g_raw; add task-k examples to the anchor set afterwards

**Transfer notes:** Brief mapping 3 (EWC-style importance). All members are SFT; whether RL-induced regression is localized in parameters is untested. If regression is format/template overfitting (K28), importance masks may not capture it.

## When it works

- Keep top 10% of fine-tuned deltas, revert the rest: original-task avg 86.1 -> 93.1 [2402.12048#s1]
- OBS compensation on kept parameters restores target performance [2402.12048#s2]
- Penalize new updates at earlier tasks' top-2% positions: BWT -20.3 -> -10.5 [2505.02486#s2]
- Freeze top-30% activation neurons from a calibration forward pass: upstream retention 98.4 [2608.11758#s1]
- Recompute the mask before each task in a sequence [2608.11758#s2]
- Project gradients out of an anchor-gradient subspace to keep a fragile task [2601.18255#s1]

## When it fails

- Target performance is traded away and some original tasks still drop [2402.12048#s1, 2608.11758#s1]
- Larger penalty trades acquisition for retention [2505.02486#s2]
- Blocks positive backward transfer; tiny gain on a single run [2601.18255#s1]
- Freezing MLP blocks still collapses upstream; tested only on SFT, not RL [2608.11758#s1]
- Compensation uses no original-task data, so preservation is not guaranteed [2402.12048#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2402.12048#s1` — Model Tailor: Mitigating Catastrophic Forgetting in Multi-modal Large Language Models (https://arxiv.org/html/2402.12048v1; Sec 4.3 Eq 8-11; Tables 1-2; Fig 5)
- `2402.12048#s2` — Model Tailor: Mitigating Catastrophic Forgetting in Multi-modal Large Language Models (https://arxiv.org/html/2402.12048v1; Thm 4.3, Thm 4.5 (proofs Appendix C, not read); empirical: Table 3; Sec 5.4; Appendix D.1)
- `2505.02486#s2` — SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning (https://arxiv.org/html/2505.02486v1; Sec 3.2-3.3; Table 1; Table 2; Table 4; Table 5; Appendix G)
- `2608.11758#s1` — AWARe: Mitigating Catastrophic Forgetting via Activation-Weighted Adaptive REtention (https://arxiv.org/html/2608.11758v1; Sec 3.2-3.4; Tables 1-2; Tables 3-6; App. A-B)
- `2608.11758#s2` — AWARe: Mitigating Catastrophic Forgetting via Activation-Weighted Adaptive REtention (https://arxiv.org/html/2608.11758v1; App. A, Tables 7-8)
- `2601.18255#s1` — Beyond Retention: Orchestrating Structural Safety and Plasticity in Continual Learning for LLMs (https://arxiv.org/html/2601.18255v1; Table 1; Fig 2; Sec 5.3)
