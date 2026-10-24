# Put visual information back into the context mid-generation

Card `K42-reinject-visual-tokens-during-decoding` · status: candidate_pattern · members: 3 signatures from 3 full-text papers

## Operation

During decoding, re-insert visual information (copy selected cached patch embeddings, re-insert the whole image at logical boundaries, crop predicted boxes and re-answer with the crops).

- `2505.18842#s1`: add linear query/key heads whose logits over image patch positions are concatenated with vocabulary logits; when a patch index is sampled, inject that patch's embedding as the next input token (point-and-copy), trained with SFT and z-loss on top-k partition
- `2509.25848#s2`: at four roughly evenly spaced logical boundaries during generation, re-insert the input image tokens (visual replay) or insert a text cue such as 'I need to see the image' (focus prompt), then continue decoding
- `2605.31096#s3`: prompt for grounded CoT, crop each predicted box plus the union (minimum enclosing) box, then re-answer with textual CoT given the global image and all crops as multi-view input

**Transfer notes:** n/a (in-domain).

## When it works

- Point-and-copy patches: MathVision-mini 23.6 -> 34.5 [2505.18842#s1]
- Visual replay at four boundaries: about +1.5 points [2509.25848#s2]
- Predicted-box crops plus global image: 80.1 -> 82.7 [2605.31096#s3]

## When it fails

- Wrong-region copying; disabling pointing falls below the backbone [2505.18842#s1]
- Marginal once trained with VAPO [2509.25848#s2]
- Individual crops alone can hurt; doubles inference [2605.31096#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2505.18842#s1` — v1: Learning to Point Visual Tokens for Multimodal Grounded Reasoning (https://arxiv.org/html/2505.18842v7; Sec 4.2, 5; Table 1; Table 2; Appendix A Table 3; Appendix B Tables 4-6)
- `2509.25848#s2` — More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models (https://arxiv.org/html/2509.25848v3; Sec 3.2, Fig 3; Table 3; Appendix B.7 Table 9)
- `2605.31096#s3` — iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning (https://arxiv.org/html/2605.31096v1; Sec 3.4; Table 4)
