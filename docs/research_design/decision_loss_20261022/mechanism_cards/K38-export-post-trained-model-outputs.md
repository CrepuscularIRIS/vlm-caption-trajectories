# Reuse an RL-post-trained model's outputs or encoder to train another model

Card `K38-export-post-trained-model-outputs` · status: candidate_pattern · members: 3 signatures from 3 full-text papers

## Operation

After RL or preference post-training, export the model's captions, reasoning text or vision encoder as training material for a different model.

- `2509.22647#s3`: Annotate a large image pool with the CapRL-trained captioner and use the captions in the further-pretraining stage of a ShareGPT4V-style three-stage pipeline
- `2510.11369#s2`: contrastively fine-tune the CLIP image encoder (text encoder frozen) to match the RL model's reasoning text, reduce embeddings by PCA, run k-means within score buckets to obtain basis vectors with scores, then fine-tune the bases on image-score pairs and score by softmax-weighted cosine similarity
- `2510.16333#s2`: attach the encoder to a small LLM, pretrain the MLLM, post-train it with DPO, detach and freeze the encoder (optionally keep the first projector layer), and reuse it as the vision backbone of a new MLLM

**Transfer notes:** n/a (in-domain).

## When it works

- RL-trained captioner's captions as pretraining data beat DenseFusion-1M [2509.22647#s3]
- CLIP scorer aligned to RL reasoning text nearly matches the MLLM at 3.4% inference time [2510.11369#s2]
- DPO-trained encoder reused in a new MLLM beats a larger encoder [2510.16333#s2]

## When it fails

- Evaluated on 12 benchmarks only [2509.22647#s3]
- Bounded by the CLIP encoder [2510.11369#s2]
- Gain shrinks with full-parameter training of the new MLLM [2510.16333#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2509.22647#s3` — CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning (https://arxiv.org/html/2509.22647v1; Sec 4.1; Table 1; Table 2; Figure 4)
- `2510.11369#s2` — Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment (https://arxiv.org/html/2510.11369v2; Sec 4; Table 2; Table 4; Fig 6; Table A.5)
- `2510.16333#s2` — RL makes MLLMs see better than SFT (https://arxiv.org/html/2510.16333v2; Sec 6 Table 2; Appendix B.7 Tables H-J)
