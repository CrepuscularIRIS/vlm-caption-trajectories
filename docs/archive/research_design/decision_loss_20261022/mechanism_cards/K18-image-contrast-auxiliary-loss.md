# Add a loss on the contrast between image-conditioned and image-removed distributions

Card `K18-image-contrast-auxiliary-loss` · status: candidate_pattern · members: 6 signatures from 3 full-text papers

## Operation

Add an objective defined on with-image vs masked/no-image distributions: maximize the policy's own divergence from its masked-image self, steer the student toward the teacher's visual log-ratio, or penalize/anchor image-free likelihood.

- `2507.06448#s1`: Add to the GRPO/DAPO objective a weighted term that maximizes KL(pi(o|q,I) || pi(o|q,I_masked)) over rollout tokens, computed with one extra forward pass on the masked image, so responses become less probable when visual content is removed
- `2507.06448#s2`: Add a Double Entropy Loss that penalizes the entropy of both the original-image and masked-image policy over rollout tokens, to regularize the maximized perception KL
- `2606.00564#s1`: construct a per-token visual target q*_T proportional to p_S(.|x) * exp(log q_T(.|I,x) - log q_T(.|x)) (softmax-normalized), and add gamma * KL(p_S(.|I,x) || q*_T) to the standard KL(p_S(.|I,x) || q_T(.|I,x)), rescaling the total by a fixed constant eta(gamma) so the gradient norm matches the standard gradient
- `2606.00564#s3`: optimize (1-alpha) * GRPO clipped surrogate (no KL penalty) - alpha * l_VGS-LP on the same group rollouts, alpha=0.3
- `2605.25036#s1`: add alpha*|log pi_theta(y|x) - log pi_ref(y|x)| (sequence-level L1 on text-only likelihood drift) to the VIT MLE loss
- `2605.25036#s2`: add gamma * -log sigmoid(beta * log pi_ref(y|x)/pi_theta(y|x)) to DPO+margin loss, pushing the image-free likelihood of responses below the reference

**Transfer notes:** n/a (in-domain). Related to K17: same signal, used as a loss rather than a weight.

## When it works

- Maximized KL to the 60%-patch-masked image: 4.4-17.5% relative gain over GRPO/DAPO [2507.06448#s1]
- Double entropy loss is needed to keep that KL term from collapsing [2507.06448#s2]
- Distillation toward the teacher's image/no-image log-ratio: 46.10 vs 43.74 standard OPD (2B) [2606.00564#s1]
- Combined with GRPO: 47.20 vs 45.41 GRPO+OPD; prevents length explosion [2606.00564#s3]
- L1 anchor on text-only likelihood drift during visual instruction tuning: hallucination count 155 -> 121 [2605.25036#s1]
- Pushing image-free likelihood below reference in DPO: MMHal 2.42 -> 2.91 [2605.25036#s2]

## When it fails

- Unbounded KL is hacked: collapse at weight 0.04 or masking 1.0, more readily in larger models [2507.06448#s1, 2507.06448#s2]
- No gain when the teacher ignores the image or on low visual-dependency tasks [2606.00564#s1]
- Alpha-sensitive; needs the pre-VIT checkpoint [2605.25036#s1]
- On Qwen2.5-VL-3B not above the base on MMHal; CHAIR worse [2605.25036#s2]
- Only a 2B student [2606.00564#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2507.06448#s1` — Perception-Aware Policy Optimization for Multimodal Reasoning (https://arxiv.org/html/2507.06448v5; Table 1, Table 2; Fig 1, 4, 5, 6; Sec 2.2, 5.1, 5.2)
- `2507.06448#s2` — Perception-Aware Policy Optimization for Multimodal Reasoning (https://arxiv.org/html/2507.06448v5; Sec 5.3; Fig 7, 8; Table 8; Appendix H)
- `2606.00564#s1` — Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding (https://arxiv.org/html/2606.00564v1; Sec 3.1 Eq 5-11; Sec 4 Eq 13-15; Table 1; Appendix A; Appendix B.3 Figure 8a)
- `2606.00564#s3` — Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding (https://arxiv.org/html/2606.00564v1; Sec 8 Eq 22; Table 2; Sec 9 Figure 6b)
- `2605.25036#s1` — Language Bias in LVLMs: From In-Depth Analysis to Simple and Effective Mitigation (https://arxiv.org/html/2605.25036v1; Sec 3.2; Sec 4.1; Tables 1, 2, 6, 8, 9; Table 17)
- `2605.25036#s2` — Language Bias in LVLMs: From In-Depth Analysis to Simple and Effective Mitigation (https://arxiv.org/html/2605.25036v1; Sec 4.2; Tables 3, 4, 5, 7, 14, 15, 16)
