# Measure image use by controlled changes to the input channels

Card `K45-modality-ablation-diagnostics` · status: candidate_pattern · members: 4 signatures from 3 full-text papers

## Operation

Measure how much behaviour comes from the image by removing vision, perturbing textual priors, injecting misleading text, or tracking image-removed likelihood gains during training.

- `2510.04532#s1`: train otherwise identical agents with selected input/output modules removed (no vision, no CoT, no history, no ego, no priors) and compare nuPlan open/closed-loop scores
- `2510.04532#s3`: add a small lateral velocity offset delta*v_ego to the ego state (or mirror the lateral history) and measure Mean Final Deviation and Relative Intervention Degree, plus check whether CoT and plan direction contradict
- `2602.12506#s1`: Inject a procedurally generated misleading caption (Wrong-Caption), seed the CoT with a wrong reasoning prefix (Wrong-Think), or suppress reasoning with an empty think tag (Stop-Think), optionally with a disclaimer or 'but I think' marker, then measure accuracy, letter entropy, P(correct letter), and answer-CoT consistency via an LLM judge
- `2605.25036#s3`: track the image-conditioned and image-removed log-likelihood ratios against the reference throughout training to quantify how much improvement is text-driven

**Transfer notes:** n/a (in-domain diagnostic).

## When it works

- A no-image, no-CoT driving agent matches the CoT agent; removing priors collapses it [2510.04532#s1]
- Small ego-velocity offsets move plans by up to 15 m while CoT stays scene-correct [2510.04532#s3]
- Misleading captions or wrong think-prefixes drive RL-tuned VLMs' correct-letter probability near zero [2602.12506#s1]
- Image-removed likelihood gains track image-conditioned gains during VIT [2605.25036#s3]

## When it fails

- 300-scenario probe; manual inversion check [2510.04532#s3]
- Faithfulness measured as external consistency only [2602.12506#s1]
- Text-only gain may be format learning, not image neglect [2605.25036#s3]
- 7B LoRA SFT only [2510.04532#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2510.04532#s1` — More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models (https://arxiv.org/html/2510.04532v2; Table 2; Sec 4.2; Appendix A.15.1 Table 11)
- `2510.04532#s3` — More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models (https://arxiv.org/html/2510.04532v2; Sec 3.4, 4.4-4.5; Table 4; Fig 2)
- `2602.12506#s1` — On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs (https://arxiv.org/html/2602.12506v3; Sec 2.1-2.4; Fig 2-7; Tables 1-6)
- `2605.25036#s3` — Language Bias in LVLMs: From In-Depth Analysis to Simple and Effective Mitigation (https://arxiv.org/html/2605.25036v1; Sec 3.2; Fig 3; Appendix D.1, Fig 8)
