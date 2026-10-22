# Fine-tune a VLM reward model to be invariant to task-preserving transforms

Card `K55-vlm-reward-model-invariance-tuning` · status: case · members: 2 signatures from 1 full-text papers

## Operation

During online RL, tune LoRA in a CLIP-style reward model so rewards vary little across task-preserving augmentations and more across negatives, optionally regularized toward Lipschitz proportionality with state distance.

- `2608.03875#s1`: after each rollout, compute VLM rewards for each observation's positive augmentations and contrastive negatives, and backpropagate a loss that minimizes reward variance across positives while maximizing variance across negatives (Contrastive Augmentation Loss), updating only LoRA
- `2608.03875#s2`: over a window of consecutive states, normalize rewards and states by their window means and penalize deviation from proportionality between reward differences and L2 state distances (soft Reward Lipschitz Regularization)

**Transfer notes:** Cross-domain (VLM as reward for control). For VLM post-training it would apply to learned reward models/judges when task-preserving image transforms are known.

## When it works

- Faster convergence and lower EPIC distance to GT reward [2608.03875#s1]
- Lipschitz regularization helps CartPole/MountainCar [2608.03875#s2]

## When it fails

- Cannot fix a fundamentally misaligned VLM; hand-chosen augmentations; doubles time [2608.03875#s1]
- Inapplicable with symmetric states or inaccessible state [2608.03875#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2608.03875#s1` — Enhancing VLM Reward Models Through Structure-Aware Fine-Tuning (https://arxiv.org/html/2608.03875v1; Sec 3.1.1; Sec 4.2-4.4; Fig. 6; Table 2; Table 3; App. B.2-B.3, B.7)
- `2608.03875#s2` — Enhancing VLM Reward Models Through Structure-Aware Fine-Tuning (https://arxiv.org/html/2608.03875v1; Sec 3.1.2; Fig. 6; App. B.4-B.5)
