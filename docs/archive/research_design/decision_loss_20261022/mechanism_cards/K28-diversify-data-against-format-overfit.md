# Mix diverse-format or general data into narrow post-training

Card `K28-diversify-data-against-format-overfit` · status: candidate_pattern · members: 4 signatures from 4 full-text papers

## Operation

Mix general or format-diverse data into narrow fine-tuning (general instruction data, rewritten answer styles, replayed general data with adaptive objective weights, batches from the other input modality).

- `2603.14493#s2`: mix diverse instruction/task data (e.g., 50% LLaVA-665K) into the task-specific fine-tuning set
- `2505.02486#s1`: convert X% of a task's samples (default 20%) equally into four alternative answer styles with matching response-format prompts, keeping total sample count fixed, so every task's training data spans all five answer styles
- `2510.21978#s1`: replay general-domain data alongside reasoning data and, at each step, set the weight of each objective to K*softmax((c_k+i_k)/T), where c_k is the ratio of previous- to current-window mean loss and i_k is window std over the sum of means, so saturated, stable objectives are down-weighted
- `2509.14480#s3`: warm up with 30 GRPO steps on simplified tasks, then alternate batches of math, text-only Retail, and speech-user Retail rollouts

**Transfer notes:** Brief mapping 3 (replay) as practised in VLM post-training. Supports the brief's alternative that regression can be a format shift rather than parameter-level forgetting.

## When it works

- 50% LLaVA-665K keeps target accuracy and recovers ImageWikiQA; forgetting is template overfitting [2603.14493#s2]
- Rewriting 20% of each task into other answer styles: MAA 39.5 -> 59.7 [2505.02486#s1]
- Replay with weights from loss-window statistics: LISA 67.2 vs 57.6 reasoning-only [2510.21978#s1]
- Alternating speech-user batches keeps speech understanding (14.8 -> 37.4) [2509.14480#s3]

## When it fails

- Classification-style mix data gives little benefit [2603.14493#s2]
- Small cost on just-learned tasks [2505.02486#s1]
- VizWiz below coreset; single runs [2510.21978#s1]
- Still lags text agents [2509.14480#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2603.14493#s2` — Fine-tuning MLLMs Without Forgetting Is Easier Than You Think (https://arxiv.org/html/2603.14493v1; Sec 4.2-4.3; Fig 3; Fig 4)
- `2505.02486#s1` — SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning (https://arxiv.org/html/2505.02486v1; Sec 2; Table 1; Table 2; Table 3; Fig 4; Appendix F)
- `2510.21978#s1` — Beyond Reasoning Gains: Mitigating General-Capability Forgetting in Large Reasoning Models (https://arxiv.org/html/2510.21978v2; Sec 4 Eq 1; Tables 1-2; Figs 4-7)
- `2509.14480#s3` — Process-Supervised Reinforcement Learning for Interactive Multimodal Tool-Use Agents (https://arxiv.org/html/2509.14480v1; Sec 4.3-4.4; Table 3)
