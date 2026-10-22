# Distil a privileged-state teacher into an observation-only student

Card `K01-privileged-teacher-student-distillation` · status: candidate_pattern · members: 9 signatures from 8 full-text papers

## Operation

Train a teacher on training-only privileged state, then supervise an observation/history-only student to match the teacher's actions and/or latent on states the student itself visits.

- `2406.19972#s1`: train a state-based teacher with RL, then distil it into the observation-only student by DAgger: roll out a mixture policy beta_t*teacher + (1-beta_t)*student with exponentially decaying beta, label every visited state with the teacher's action, and regress the student on these labels with MSE
- `2412.10628#s1`: train a teacher policy per task with RL on privileged state using a terrain curriculum, then distil it by supervised learning into an observation-only student policy that maps depth + pose to joint angles, and deploy the student zero-shot on the real robot
- `2607.21960#s1`: train a recurrent teacher with PPO on privileged state under per-episode viscosity domain randomization, then distill it into a recurrent student by minimizing MSE between teacher and student actions on sensor-only observations
- `2407.04224#s1`: encode each privileged source with a separate encoder (force, terrain, state) and concatenate their latents as the teacher's context, then train the student's history encoder and policy with an MSE loss on both the teacher's action and the concatenated latent while rolling out the student
- `2609.20982#s1`: train a teacher VAE encoder and PPO policy on the privileged attack-aware latent, then train an LSTM temporal-VAE student encoder by regressing to the teacher latent, matching the policy's actions from both latents, and an attack-type classification loss; reuse the policy unchanged
- `2503.08997#s1`: append the privileged embedding as the last token so only the teacher head can attend to it; train in one phase with PPO loss plus a next state-action prediction loss on the proprioceptive tokens plus an imitation loss pulling student actions toward teacher actions; at deployment drop the privileged token and use the student head
- `2503.08997#s2`: resample a per-agent mask frequently so that a fraction (mix ratio) of agents act with teacher actions and the rest with student actions, and train both heads jointly with PPO on the mixed trajectories
- `2412.00985#s1`: learn a decoding function g_h(s_{h-1}, a_{h-1}, o_h) -> s_h by supervised counting on privileged-state trajectories rolled out with the expert, then compose it recursively with the state-based expert policy
- `2503.20839#s1`: predict the next-step student latent from current student latent and action via a forward-dynamics head, and apply a triplet loss with the teacher's next-state latent as anchor, that prediction as positive, and a latent from another parallel agent with different randomized environment parameters as negative; update critic encoder with value+triplet loss and student encoder with triplet loss only, blocking actor gradients

**Transfer notes:** Brief mapping 2 (privileged information). Target: a teacher that sees GT boxes/OCR/region text, a pixel-only VLM student. Must hold: the evidence the teacher uses is recoverable from the student's pixels at its resolution (deterministic-filter analogue), and the teacher is queried on student rollouts. Breaks when the student cannot perceive what the teacher uses (imitation gap) or when the privileged text encodes the answer. VLM-side instances are self-distillation (K48) rather than separate RL teachers.

## When it works

- Teacher labels are collected on states visited by a teacher/student mixture (DAgger), not offline teacher traces: 74.8% vs 15.3% success for offline BC [2406.19972#s1]
- Hidden quantities are inferable from the student's history (recurrent student infers viscosity; history encoder regresses teacher latent) [2607.21960#s1, 2609.20982#s1, 2407.04224#s1]
- Separate encoders per privileged source make the latent target usable on hardware [2407.04224#s1]
- Joint one-phase training with imitation plus PPO and an intermediate teacher/student rollout mix ratio [2503.08997#s1, 2503.08997#s2]
- Deterministic-filter condition (state decodable from previous state, action, observation) gives provable near-expert value [2412.00985#s1]
- A contrastive (triplet) alignment to the teacher's next-state latent, instead of direct regression, let the student exceed the teacher's return [2503.20839#s1]
- Privileged map/joint state in simulation, depth student deployed zero-shot on a real hexapod [2412.10628#s1]

## When it fails

- Imitation gap: when the state stays uncertain given the full history, the distilled policy is suboptimal even with infinite data [2412.00985#s1]
- Student stays 11-19 points below the teacher; unseen geometry 20% success [2406.19972#s1]
- Fails when the decisive cue leaves the camera view (last stair) and the student assumes flat ground [2412.10628#s1]
- Rollout mix extremes fail: teacher-only rollouts give a poor student, no teacher drops to PPO level [2503.08997#s2, 2503.08997#s1]
- Single-encoder coupling is unstable on hardware although simulation did not show it [2407.04224#s1]

## Counterexamples

- Baseline RL given the same privileged information but without the encoder/monitor performs like RL without it, so privileged access alone did not help [2609.20982#s1]

## Sources

- `2406.19972#s1` — HumanVLA: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid (https://arxiv.org/html/2406.19972v2; Sec 3.2 'Behavior Cloning'; Appendix E.2.5; Table 3; Table 4)
- `2412.10628#s1` — Versatile Locomotion Skills for Hexapod Robots (https://arxiv.org/html/2412.10628v1; Sec III, Tables III-V, Sec IV)
- `2607.21960#s1` — Adaptive Undulatory Locomotion of Snake-like Robots in Dynamic Viscous Environments via Deep Reinforcement Learning (https://arxiv.org/html/2607.21960v1; Sec 4.1-4.2; Fig. 3a-b; Fig. 4; Sec 6.1-6.2)
- `2407.04224#s1` — PA-LOCO: Learning Perturbation-Adaptive Locomotion for Quadruped Robots (https://arxiv.org/html/2407.04224v2; Sec IV 'Teacher Policy Architecture', Eq 1; Table V; Figs 5-6)
- `2609.20982#s1` — ASGARD: Action-Space Guard for UAV Resilience via Reinforcement Learning (https://arxiv.org/html/2609.20982v1; Sec IV-B, IV-C; Fig. 2; Sec V-B)
- `2503.08997#s1` — Unified Locomotion Transformer with Simultaneous Sim-to-Real Transfer for Quadrupeds (https://arxiv.org/html/2503.08997v2; Sec IV-A, IV-C, Table IV, Fig 3)
- `2503.08997#s2` — Unified Locomotion Transformer with Simultaneous Sim-to-Real Transfer for Quadrupeds (https://arxiv.org/html/2503.08997v2; Sec IV-B Eq 14; Sec V-A Fig 2)
- `2412.00985#s1` — Provable Partially Observable Reinforcement Learning with Privileged Information (https://arxiv.org/html/2412.00985v3; Definition 3.2; Lemma 4.4; Theorems 4.5-4.6; Table 2)
- `2503.20839#s1` — TAR: Teacher-Aligned Representations via Contrastive Learning for Quadrupedal Locomotion (https://arxiv.org/html/2503.20839v2; Sec III-B2-B3; Algorithm 1; Fig 4; Fig 5; Sec IV-A to IV-C)
