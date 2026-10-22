# Choose the next view by confidence-weighted Fisher information gain

Card `K54-fisher-information-next-best-view` · status: case · members: 3 signatures from 1 full-text papers

## Operation

Score candidate views by the log-det increase of a confidence-weighted Fisher information of the scene model, normalize gains per output, optionally restrict to a target object, and take the argmax.

- `2602.08266#s1`: Score each candidate by the increase in log-det of a block-diagonal confidence-weighted Fisher information H = J^T C J, where each Gaussian's weight is max object probability^-alpha_obj times opacity^-alpha_opa, then pick the argmax view
- `2602.08266#s2`: Normalize each output's information gain by its mean gain over current training views (so a training view has unit gain) and sum RGB + beta_d depth + beta_o mask
- `2602.08266#s3`: Replace the max-class confidence by the target-object probability for Gaussians whose argmax class is the target and zero weight elsewhere, then run the same information-gain NBV

**Transfer notes:** Brief mapping 1 source (cross-domain). Target: crop/zoom choice by expected reduction of answer uncertainty. Must hold: the VLM's answer distribution behaves like a belief whose curvature can be computed and crops change it; here the belief is an explicit 3DGS map with Jacobians, which a VLM lacks. Breaks under miscalibrated answer entropy.

## When it works

- Depth error down by up to 77% vs FisherRF and others [2602.08266#s1]
- Gain normalization by training-view gain improves further [2602.08266#s2]
- Target-object restriction: further 25.6% for the target [2602.08266#s3]

## When it fails

- Mask errors propagate to view choice [2602.08266#s1]
- Centre objects share visibility across views; real-robot results qualitative [2602.08266#s3]

## Counterexamples

- None found among the read signatures.

## Sources

- `2602.08266#s1` — Informative Object-centric Next Best View for Object-aware 3D Gaussian Splatting in Cluttered Scenes (https://arxiv.org/html/2602.08266v1; Sec III-B; Table I-III; Abstract)
- `2602.08266#s2` — Informative Object-centric Next Best View for Object-aware 3D Gaussian Splatting in Cluttered Scenes (https://arxiv.org/html/2602.08266v1; Sec III-B3; Table III)
- `2602.08266#s3` — Informative Object-centric Next Best View for Object-aware 3D Gaussian Splatting in Cluttered Scenes (https://arxiv.org/html/2602.08266v1; Sec III-C; Sec IV-B; Table IV; Fig 6, Fig 8)
