# Add human-annotated short-answer VQA to instruction tuning before alignment

Card `K34-human-vqa-in-instruction-tuning` · status: candidate_pattern · members: 2 signatures from 2 full-text papers

## Operation

Mix human-annotated short-answer VQA/MCQ/grounded-caption data into instruction tuning ahead of preference alignment, and avoid augmentations that break image-text correspondence.

- `2309.14525#s3`: augment GPT-4-synthesised LLaVA instruction data with human-annotated VQA/MCQ/grounded-caption data converted into multi-round conversations
- `2312.00849#s3`: post-train on human-annotated VQAv2 before/with preference learning and exclude random image cropping from data augmentation to remove image-text mismatch

**Transfer notes:** n/a (in-domain).

## When it works

- MMBench 38.7 -> 52.1 and POPE 76.0 -> 82.7 [2309.14525#s3]
- Removing VQAv2 tuning raises hallucination 55.5 -> 65.1; random crop raises it to 59.6 [2312.00849#s3]

## When it fails

- Single-dataset additions sharply lower LLaVA-Bench [2309.14525#s3]
- Crop ablation lowers object hallucination while raising other types [2312.00849#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2309.14525#s3` — Aligning Large Multimodal Models with Factually Augmented RLHF (https://arxiv.org/html/2309.14525v1; Sec 2.2; Table 5; Sec 3.5)
- `2312.00849#s3` — RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback (https://arxiv.org/html/2312.00849v2; Sec 3.2; Table 3 'w/ IT-VQA only', 'w/ untrust aug.')
