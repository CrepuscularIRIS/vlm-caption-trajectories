# Keep training items whose answer depends on the visual input

Card `K15-visual-dependency-data-filter` · status: candidate_pattern · members: 2 signatures from 2 full-text papers

## Operation

Score each item by the contrast between model performance with full visual input and with reduced or no visual input, and keep items where the visual input matters.

- `2509.22647#s2`: Keep only questions whose image-conditioned accuracy is high and question-only accuracy is low (thresholded), discarding questions answerable from text priors
- `2502.12081#s1`: score each video-text pair by temporal perplexity TPL = NLL(text | one frame) - NLL(text | all frames) and use it to select or mix training data

**Transfer notes:** n/a (in-domain).

## When it works

- MCQs kept only if answerable with the image and not without it: 48.5 vs 47.4 for leaking data [2509.22647#s2]
- Video-text pairs scored by NLL(one frame) - NLL(all frames); higher-score subsets train better models [2502.12081#s1]

## When it fails

- Leaking data degrades but does not collapse training; generated answer keys unverified [2509.22647#s2]
- Correlational; low score can reflect near-static video rather than a bad caption [2502.12081#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2509.22647#s2` — CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning (https://arxiv.org/html/2509.22647v1; Sec 3.1 QA curation (Eq. 4); Appendix B Table 6; Appendix E; Sec 4.3 Table 5)
- `2502.12081#s1` — Unhackable Temporal Rewarding for Scalable Video MLLMs (https://arxiv.org/html/2502.12081v1; Sec 2.2 Eq 6, Figure 2; Sec 4.5 Figure 4; Appendix E.2 Table 12)
