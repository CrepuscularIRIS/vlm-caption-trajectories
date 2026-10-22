# Score predicted boxes against ground truth with matching and anti-hacking terms

Card `K08-rule-based-box-rewards` · status: candidate_pattern · members: 4 signatures from 4 full-text papers

## Operation

Reward predicted boxes by IoU/matching against GT boxes in GRPO, adding terms that block box-size exploitation (confidence, Hungarian matching, size term, IoU threshold schedule).

- `2503.01785#s1`: Sort predicted boxes by confidence, match each to GT boxes with an IoU threshold (unmatched/below-threshold = IoU 0), reward = mean IoU + confidence reward (high confidence rewarded on matched boxes, low confidence rewarded on unmatched ones) + format reward; normalize within GRPO groups and update with KL to reference
- `2504.07954#s1`: solve a bipartite matching (Hungarian algorithm) between predicted and GT objects maximizing the per-pair discriminative reward (IoU for boxes, Euclidean distance for points), then average matched rewards and add a +/-1 format reward; for detection, combine F1 count reward, IoU location reward, binary class reward and a missing-detection penalty
- `2505.15810#s1`: add a box-size reward term that scores how closely the predicted box width/height match the ground-truth box, combined with hit-based and IoU-based rewards (weights 0.25 and 0.125 for auxiliary terms)
- `2505.18129#s1`: replace a fixed IoU threshold with a three-stage step-based schedule that raises the threshold from 0.85 (first 10% of steps) to 0.95 (next 15%) to 0.99 (rest), rewarding boxes only above the current threshold

**Transfer notes:** n/a (in-domain). Reusable as the GT-region reward inside crop-tool RL only where GT boxes exist.

## When it works

- Few-shot detection: COCO 1-shot mAP 19.6 -> 33.6 vs SFT 19.5 [2503.01785#s1]
- Bipartite matching for multi-object rewards: COCO AP 31.9 vs 23.5 with sequential matching [2504.07954#s1]
- A box-size term counters opposite size hacking of hit (shrink) and IoU (enlarge) rewards [2505.15810#s1]
- Step-staged IoU threshold (0.85 -> 0.95 -> 0.99) beats fixed thresholds [2505.18129#s1]

## When it fails

- Box-size reward alone fails to produce valid outputs [2505.15810#s1]
- Adaptive success-rate controller did not beat a hand-set schedule [2505.18129#s1]
- Still far below expert detectors (31.9 vs 42.0 AP) [2504.07954#s1]
- 7B gains smaller than 2B; some rare classes stay near 0 AP [2503.01785#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2503.01785#s1` — Visual-RFT: Visual Reinforcement Fine-Tuning (https://arxiv.org/html/2503.01785v1; Sec 3.2.1; Tables 3, 4, 5, 7, 8; Sec 4.3, 4.5)
- `2504.07954#s1` — Perception-R1: Pioneering Perception Policy with Reinforcement Learning (https://arxiv.org/html/2504.07954v1; Sec 4.2; Sec 4.3 Reward Setting; Table 3(b); Table 5 rows 1-2; Table 6)
- `2505.15810#s1` — GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents (https://arxiv.org/html/2505.15810v2; Sec 3.2, Table 1, Figure 3, Figure 4)
- `2505.18129#s1` — One RL to See Them All: Visual Triple Unified Reinforcement Learning (https://arxiv.org/html/2505.18129v3; Sec 4.5; Fig 2(a); Appx J, L, M)
