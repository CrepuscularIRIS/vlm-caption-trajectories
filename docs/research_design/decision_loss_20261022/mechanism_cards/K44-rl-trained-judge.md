# Train an MLLM judge or verifier with RLVR on verdict labels

Card `K44-rl-trained-judge` · status: candidate_pattern · members: 3 signatures from 3 full-text papers

## Operation

Train the judge/critic/verifier with GRPO/DAPO, without SFT, using preference, verdict or ranking labels as the verifiable reward.

- `2509.00676#s1`: Strip rationales and metric prompts, pose the pair as a choose-1/2/tie question, and run GRPO on a base or reasoning VLM with reward 0.9*exact-match-preference + 0.1*think/boxed format reward, without any SFT
- `2606.02578#s2`: train the judge with GRPO (5 rollouts, KL 0.01) on reward R = R_format x (1 - Levenshtein(predicted permutation, r_c>r_p>r_p+r)/3), giving 1, 2/3, 1/3 or 0 per permutation
- `2510.13804#s2`: train with DAPO separately on single atomic-skill datasets (object, attribute, spatial, maze) and measure transfer to all ViVerBench tasks

**Transfer notes:** n/a (in-domain).

## When it works

- Critic data alone lifts policy benchmarks (26-benchmark avg 54.3 -> 57.4) [2509.00676#s1]
- Permutation-distance reward over three graded candidates: judge Pearson 0.404 -> 0.466 [2606.02578#s2]
- Object- or spatial-only verdict RL transfers to alignment tasks [2510.13804#s2]

## When it fails

- Policy performance peaks then declines; agent ability not restored [2509.00676#s1]
- LLaVA-Critic stays higher on pairwise metrics [2606.02578#s2]
- Integrative reasoning tasks do not benefit [2510.13804#s2]

## Counterexamples

- None found among the read signatures.

## Sources

- `2509.00676#s1` — LLaVA-Critic-R1: Your Critic Model is Secretly a Strong Policy Model (https://arxiv.org/html/2509.00676v1; Table 3; Sec 3.1.2; Appendix .1-.2 (Tables 9-10, referenced, not read))
- `2606.02578#s2` — Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling (https://arxiv.org/html/2606.02578v1; Sec 3.4 Eq 5-8; Table 1; Table 2; Table 3 rows 3-4; Appendix C.2; Appendix H.2)
- `2510.13804#s2` — Generative Universal Verifier as Multimodal Meta-Reasoner (https://arxiv.org/html/2510.13804v1; Sec 4.2; Fig 3; Fig 4)
