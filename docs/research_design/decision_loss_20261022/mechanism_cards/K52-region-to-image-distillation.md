# Generate QA on micro-crops and re-anchor it to the full image with a training-only box hint

Card `K52-region-to-image-distillation` · status: case · members: 2 signatures from 1 full-text papers

## Operation

Generate perception QA with consensus teachers on object micro-crops, re-anchor the question to the full image by drawing the crop box, drop items the student already solves, and RL the student.

- `2602.11858#s1`: Generate perception questions and consensus answers on micro-crops, then re-anchor each question to the full image by drawing the crop box on the image and appending a spatial constraint, and drop items the base student already answers correctly in >2 of 4 trials
- `2602.11858#s2`: Present the region either as a box drawn on the image, as box coordinates in the question text, or not at all, and train the same student with RL

**Transfer notes:** Instance of brief mapping 2: the box is privileged at training and absent at test.

## When it works

- R2I with box overlay 70.89 vs 66.94 direct synthesis; 74K data beats much larger tool-RL corpora [2602.11858#s1]
- Box drawn in the image beats box coordinates in text or no box [2602.11858#s2]

## When it fails

- Spatial and multi-object perception excluded; counting hard; in-house benchmark shares construction [2602.11858#s1]
- Attention-alignment mechanism posited, not tested [2602.11858#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2602.11858#s1` — Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception (https://arxiv.org/html/2602.11858v2; Sec 3.1; Table 2, Table 3, Table 6; Appendix 8.1)
- `2602.11858#s2` — Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception (https://arxiv.org/html/2602.11858v2; Sec 4.3 Table 6; Sec 5.2 Table 8)
