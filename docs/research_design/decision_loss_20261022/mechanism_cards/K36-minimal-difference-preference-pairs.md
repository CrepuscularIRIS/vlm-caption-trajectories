# Build preference pairs that differ only in the hallucinated content

Card `K36-minimal-difference-preference-pairs` · status: candidate_pattern · members: 3 signatures from 2 full-text papers

## Operation

Construct preference pairs that share all but the hallucinated segments (human correction of a sampled response; GT prefix completed by the image-masked policy as rejected) and optionally up-weight the differing tokens.

- `2312.00849#s1`: ask humans to correct hallucinated segments in a sampled response, producing a factually corrected response y_w and the original y_l as an incremental segment-level preference pair with a token mask of corrected vs unchanged segments
- `2312.00849#s2`: replace the DPO sequence log-prob with a normalized weighted sum in which tokens of corrected segments are multiplied by a weight gamma (=5) and unchanged tokens by 1, divided by a length normalizer, then apply the standard DPO loss (beta=0.5)
- `2407.02477#s1`: split each SFT ground-truth sentence at a random position, keep the prefix, let the image-attention-masked policy complete the sentence (each sentence conditioned on the GT preceding sentences); flip leading yes/no with prob 0.5 for short answers; if the cosine similarity of the result to the GT exceeds a threshold, resample a new mask and split, up to a max number of iterations, then fall back to unguided masked generation; pair GT (chosen) vs this output (rejected) and train with DPO, offline or regenerated online

**Transfer notes:** n/a (in-domain).

## When it works

- 200 segment-corrected pairs match about ten times more ranking data [2312.00849#s1]
- Up-weighting corrected tokens in DPO lowers some hallucination types [2312.00849#s2]
- Image-masked self-generated rejects beat GPT-4V-corrupted POVID pairs [2407.02477#s1]

## When it fails

- Needs paid human correction; scaling beyond 1.4k not shown [2312.00849#s1]
- Small and non-uniform gain (number hallucination worse) [2312.00849#s2]
- Text-judge MMHal regresses; one model family [2407.02477#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2312.00849#s1` — RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback (https://arxiv.org/html/2312.00849v2; Sec 2; Sec 4.3 'Fine-grained correctional human feedback enables better learning efficiency'; Fig 2)
- `2312.00849#s2` — RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback (https://arxiv.org/html/2312.00849v2; Sec 3.1 Eq 5; Table 1; Table 3 'w/ vanilla DPO')
- `2407.02477#s1` — Understanding Alignment in Multimodal LLMs: A Comprehensive Study (https://arxiv.org/html/2407.02477v1; Sec 3.4, Table 2, Table 6, Sec 5.4)
