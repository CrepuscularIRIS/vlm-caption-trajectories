# Turn a verifier's critique into a local repair when it flags failure

Card `K43-verifier-triggered-local-repair` · status: candidate_pattern · members: 2 signatures from 2 full-text papers

## Operation

When a verifier's verdict or confidence flags a failure, convert its critique into a local repair instruction and regenerate only that segment or region.

- `2511.19900#s2`: when the step confidence falls below the threshold, have the Verifier issue a local repair instruction and have the Solver regenerate that segment, charging a repair penalty in the step reward
- `2510.13804#s3`: iteratively verify the generated image; when the verdict is false, rephrase the explanation into an edit instruction and have the same model edit the image, stopping at a true verdict or 10 steps

**Transfer notes:** One member is image generation (outside the brief's scope); kept because the operation is shared.

## When it works

- Self-repair removal costs about 2.5% average [2511.19900#s2]
- Verify-then-edit loop for generated images beats parallel Best-of-10 at about half the time [2510.13804#s3]

## When it fails

- No calibration analysis of the confidence gate [2511.19900#s2]
- Colour drift over iterative edits; some sub-scores drop [2510.13804#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2511.19900#s2` — Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning (https://arxiv.org/html/2511.19900v2; Sec 3.2 Eq. 3-4; Table 3; Fig 4)
- `2510.13804#s3` — Generative Universal Verifier as Multimodal Meta-Reasoner (https://arxiv.org/html/2510.13804v1; Sec 5.1-5.3; Tables 3-4)
