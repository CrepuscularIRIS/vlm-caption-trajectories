# Limit how much and where the model moves during post-training

Card `K27-restrict-update-scope` · status: candidate_pattern · members: 5 signatures from 4 full-text papers

## Operation

Restrict the update by small learning rate, LoRA-only training, frozen vision encoder/projector, per-module learning rates, or merging one LoRA per task.

- `2603.14493#s1`: fine-tune with a small learning rate (1e-6 instead of 1e-5) or LoRA on the LLM backbone, keeping vision encoder and projector frozen
- `2603.14493#s3`: for each task train a new LoRA adapter and merge it into the base weights before the next task (IncLoRA), or sequentially full-fine-tune all parameters with the MLLM-CL recipe (SeqFull)
- `2509.22195#s2`: restrict fine-tuning to LoRA adapters (no full-parameter update, no co-training mixture)
- `2406.03496#s3`: stage 1: train only the projector and visual learners (lr 1e-3) on image-text pairs with learner outputs added directly; stage 2: add textual learners and router and update the LLM branch with a small learning rate (2e-6) while the projector uses a larger rate (1e-5)
- `2505.18129#s3`: monitor training metrics per data source and, based on observed ViT gradient explosion and leaked image placeholder tokens, freeze the vision encoder and connector (update LLM only) and strip generated image/video special tokens from rollouts before logit recomputation

**Transfer notes:** n/a (in-domain).

## When it works

- lr 1e-6 instead of 1e-5: OOD drop -1.5 vs -33.6 pp at equal target accuracy [2603.14493#s1]
- Sequential LoRA merge beats specialised continual-learning methods without replay [2603.14493#s3]
- LoRA-only VLA fine-tuning keeps >85% of VQA scores [2509.22195#s2]
- Low LLM lr with higher projector lr balances text-only and multimodal ability [2406.03496#s3]
- Freezing ViT and connector avoids gradient explosion and detection collapse in mixed RL [2505.18129#s3]

## When it fails

- ID-image/OOD-text drop remains [2603.14493#s1]
- Comparison confounds backbone, data and method [2509.22195#s2]
- Evidence only as figure bars [2406.03496#s3]
- Shown only for one RL setup [2505.18129#s3]
- Single benchmark and task order [2603.14493#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2603.14493#s1` — Fine-tuning MLLMs Without Forgetting Is Easier Than You Think (https://arxiv.org/html/2603.14493v1; Sec 3.1-3.3; Table 1; Table 2; Fig 2)
- `2603.14493#s3` — Fine-tuning MLLMs Without Forgetting Is Easier Than You Think (https://arxiv.org/html/2603.14493v1; Sec 5.1-5.2; Table 3)
- `2509.22195#s2` — Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting (https://arxiv.org/html/2509.22195v1; Table 1; Sec 4.1)
- `2406.03496#s3` — Wings: Learning Multimodal LLMs without Text-only Forgetting (https://arxiv.org/html/2406.03496v1; Sec 3.3; Sec 4.3; Fig 5(b),(c); Appendix A)
- `2505.18129#s3` — One RL to See Them All: Visual Triple Unified Reinforcement Learning (https://arxiv.org/html/2505.18129v3; Fig 2(c); Fig 4; Appx B.1, B.2)
