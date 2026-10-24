# Use attention mass per source as a proxy for which input is used

Card `K47-attention-share-diagnostics` · status: candidate_pattern · members: 2 signatures from 2 full-text papers

## Operation

Aggregate attention weights by source token set (text before/after image, images, priors, CoT) across layers and relate them to capability changes.

- `2406.03496#s1`: for each layer sum the attention mass on text-before-image and text-after-image tokens to form two layer curves (Laws), and score attention shift as 1 minus the expected Pearson correlation between the two curves
- `2510.04532#s4`: compute proportional attention of each output sequence to each preceding source sequence at shallow, middle and final layers

**Transfer notes:** n/a (in-domain diagnostic).

## When it works

- Larger before/after-image attention shift correlates with text-only degradation across 100 MLLMs [2406.03496#s1]
- During planning, image attention falls below 2% while prior attention rises [2510.04532#s4]

## When it fails

- Correlational; strength not quantified [2406.03496#s1]
- Single agent [2510.04532#s4]

## Counterexamples

- An attention-ratio reward gives no gain where labelled claim probes do [2509.25848#s1]

## Sources

- `2406.03496#s1` — Wings: Learning Multimodal LLMs without Text-only Forgetting (https://arxiv.org/html/2406.03496v1; Sec 2.2 Eq 1 and Attention Shift formula; Fig 2(b),(c))
- `2510.04532#s4` — More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models (https://arxiv.org/html/2510.04532v2; Sec 3.3, 4.3; Table 3)
