# Train a per-token good/bad reward head and reuse its backbone delta

Card `K58-token-level-reward-model` · status: case · members: 2 signatures from 1 full-text papers

## Operation

Replace the LM head with a per-token sigmoid head trained on synthetic token labels, then optionally merge the LoRA delta back into the policy at reduced scale.

- `2410.04734#s2`: replace the LM head with a linear head mapping each response token's last hidden state to a sigmoid good/bad probability, and train it with per-token binary cross-entropy while LoRA-tuning both the multimodal projector and the decoder
- `2410.04734#s3`: after training the TLDR head, discard it, reattach the original LM head, and merge the LoRA deltas into the backbone at a reduced scale tau = alpha_infer/alpha_train

**Transfer notes:** n/a.

## When it works

- Response accuracy 83.1 vs 81.1 naive RM; faster human fixing [2410.04734#s2]
- Merging at tau=0.25 improves BLINK count/spatial [2410.04734#s3]

## When it fails

- Single flagged token vetoes the response: worse on spatial and small objects [2410.04734#s2]
- Collapses at tau=1; hallucination judged by itself [2410.04734#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2410.04734#s2` — TLDR: Token-Level Detective Reward Model for Large Vision Language Models (https://arxiv.org/html/2410.04734v2; Sec 5.1 Eq 6; Tables 1, 2, 4, 7; Table 11)
- `2410.04734#s3` — TLDR: Token-Level Detective Reward Model for Large Vision Language Models (https://arxiv.org/html/2410.04734v2; Sec 5.4; Tables 5-6)
