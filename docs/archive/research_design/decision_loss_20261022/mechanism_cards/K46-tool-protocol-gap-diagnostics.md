# Evaluate one checkpoint under two protocols that differ only in zoom access

Card `K46-tool-protocol-gap-diagnostics` · status: candidate_pattern · members: 4 signatures from 3 full-text papers

## Operation

Evaluate the same checkpoint with and without crop/zoom access (tool-free vs tool, full image vs evidence crop, grounded vs plain prompt) and decompose the gap.

- `2602.01334#s1`: Evaluate each checkpoint tool-free and tool-available, define the tool-induced gap as their accuracy difference, and split cumulative absolute drift into tool-free drift and change in tool-induced gap to compute a tool contribution ratio
- `2602.01334#s2`: Partition items by tool-free correctness and by call/no-call, and write the tool-induced gap as Call Gain + Schema Gain - Call Harm - Schema Harm, each factorized as Mass x Policy (P(call|domain)) x Quality (P(outcome|call,domain))
- `2602.11858#s3`: Compute the zooming gap as Regional-View accuracy minus Global-View accuracy per model and per perceptual dimension
- `2605.31096#s2`: evaluate the same grounded-trained model with a textual-CoT prompt instead of its trained grounded format and stratify accuracy by the IoU of its predicted box/crop

**Transfer notes:** n/a (in-domain diagnostic).

## When it works

- Over 70% of tool-RL progress is visible without tools [2602.01334#s1]
- Gap = call gain + schema gain - harms: harm falls while call gain plateaus [2602.01334#s2]
- Zooming gap: crop 63.1 vs full image 37.9 on the same questions [2602.11858#s3]
- Grounded-trained models answer as well with a plain textual prompt [2605.31096#s2]

## When it fails

- Tool-free accuracy is an operational proxy [2602.01334#s1]
- Descriptive, not causal [2602.01334#s2]
- Counting stays hard even on crops [2602.11858#s3]
- About 1-point differences, two models [2605.31096#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2602.01334#s1` — What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom (https://arxiv.org/html/2602.01334v2; Sec 3.1; Sec 4.2; Fig 2; Table 2)
- `2602.01334#s2` — What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom (https://arxiv.org/html/2602.01334v2; Sec 3.2-3.3; Eq 7-8; Fig 3-5; Table 1; Appendix A)
- `2602.11858#s3` — Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception (https://arxiv.org/html/2602.11858v2; Sec 5.1; Fig 7; Table 7)
- `2605.31096#s2` — iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning (https://arxiv.org/html/2605.31096v1; Sec 1; Table 1; Fig 2)
