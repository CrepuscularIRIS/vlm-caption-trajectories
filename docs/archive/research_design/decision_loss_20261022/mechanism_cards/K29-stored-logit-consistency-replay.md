# Anchor replayed samples to their stored logits with a strict distance

Card `K29-stored-logit-consistency-replay` · status: case · members: 2 signatures from 1 full-text papers

## Operation

Store each replay sample's pre-softmax logits when buffered and add a strict pointwise distance (L1/L2/L_inf) between current and stored logits to the replay loss.

- `2207.04998#s1`: store each buffered sample's logits alongside image and label, and on replay add beta times a distance (L1/L2/L_inf/MSE, or KL/MI/SSL loss) between current and stored logits to the ER cross-entropy objective
- `2207.04998#s2`: compare strict pointwise logit-matching losses against approximate SSL/MI-style losses under one ER framework and choose strict ones (L_inf penalizes the largest coordinate deviation, which the authors argue falls on current- or past-task logits and thus counteracts task-recency bias)

**Transfer notes:** Cross-domain (continual learning), brief mapping 3. Challenges KL-to-reference as the preservation constraint: probability-space KL was weak where strict logit anchors worked. For a VLM: store reference logits on a replay set of prior-ability prompts and penalize strict deviation during RL. Untested for generative sequence outputs; vocabulary-sized logits make storage costly.

## When it works

- S-CIFAR-100 buffer 500: 28.02 -> 42.15; better calibration and less recency bias [2207.04998#s1]
- Strict Lp/MSE beats KL, MI and SSL-style consistency as class count grows [2207.04998#s2]

## When it fails

- KL on softmax probabilities gives almost no gain; only small-scale image classification [2207.04998#s1]
- On S-CIFAR-10 all regularizers are similar; L_inf mechanism untested [2207.04998#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2207.04998#s1` — Consistency is the key to further mitigating catastrophic forgetting in continual learning (https://arxiv.org/html/2207.04998v1; Sec 3-4 Eq 2-3; Table 2; Table 3; Sec 7.2-7.4)
- `2207.04998#s2` — Consistency is the key to further mitigating catastrophic forgetting in continual learning (https://arxiv.org/html/2207.04998v1; Sec 6 (v)-(vii); Table 2; Appendix G)
