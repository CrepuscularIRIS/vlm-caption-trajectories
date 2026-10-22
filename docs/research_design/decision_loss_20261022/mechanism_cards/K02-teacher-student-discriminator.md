# Use a teacher-vs-student discriminator instead of plain imitation of a privileged teacher

Card `K02-teacher-student-discriminator` · status: candidate_pattern · members: 2 signatures from 2 full-text papers

## Operation

Train a discriminator that separates teacher from student actions/trajectories on student-visited states and use it as a reward term or to decide where to query the privileged teacher.

- `2504.10390#s1`: roll out the student, record the frozen teacher's response action at every visited state, train a discriminator (BCE + gradient penalty + L2 weight decay) to tell teacher vs student actions, and add the discriminator probability as a reward term in the student's PPO objective alongside task rewards
- `2505.09546#s1`: Roll out the student; query the privileged teacher only at states a learned teacher-vs-student discriminator scores below threshold kappa (states the teacher has not visited); aggregate those labels and train the student on imitation loss plus a term for fooling the discriminator

**Transfer notes:** Would transfer to VLM training where a GT-conditioned teacher shows behaviour a pixel-only student cannot realise; a discriminator over VLM trajectories would decide which teacher behaviour to imitate. Untested in VLMs; requires a learnable discriminator over text/tool trajectories.

## When it works

- Discriminator probability as PPO reward with teacher actions queried at student states: +26% terrain level vs latent-plus-action teacher-student [2504.10390#s1]
- Querying the teacher only at states the discriminator scores as unvisited by the teacher succeeds where DAgger fails under hidden-context aliasing [2505.09546#s1]

## When it fails

- Student remains bounded by the teacher [2504.10390#s1]
- Depends on discriminator strength: too strong increases aliasing, too weak yields off-distribution queries; 78% sim -> 66% real [2505.09546#s1]

## Counterexamples

- None found among the read signatures.

## Sources

- `2504.10390#s1` — Teacher Motion Priors: Enhancing Robot Locomotion over Challenging Terrain (https://arxiv.org/html/2504.10390v2; Sec II-C, II-E Eq 16, Sec III-B Figs 4-6)
- `2505.09546#s1` — Distilling Realizable Students from Unrealizable Teachers (https://arxiv.org/html/2505.09546v1; Algorithm 1; Sec III-A3 (Eq 4-7); Fig 4; Fig 6; Sec IV-B Q1, Q3)
