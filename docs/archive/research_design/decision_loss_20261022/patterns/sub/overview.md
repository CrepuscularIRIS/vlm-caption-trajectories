# VLM post-training sub-patterns — overview

19 sub-patterns induced by HDBSCAN (mcs=12) on embeddinggemma-300m embeddings of the four `abstract_*` fields of 879 VLM post-training papers (ICLR/ICML/NeurIPS 2023–2026); 281 papers (32.0%) unclustered. Each row is one cluster; the full card is `C##.md`. Examples panels are paper-agnostic lessons (no paper IDs).

| Cluster | Label | Parent (primary) | Secondary | n | oral | spotlight | poster | reject |
|---|---|---|---|---|---|---|---|---|
| `C00` | Label-Free Preference Pair Construction | Manufacture Contrasts by Controlled Corruption (`manufacture_contrastive_pairs`) | Substitute Cheap Verifiable Signals for Expensive Supervision | 46 | 0 | 0 | 35 | 11 |
| `C01` | Credit the Localization Subgoal | Make Intermediate Steps Explicit and Score Them (`reward_explicit_intermediates`) | Decompose Into Separately Verifiable Parts | 34 | 0 | 1 | 23 | 10 |
| `C02` | Isolate Weak Sub-Skill, Supervise Intermediates | Diagnose the Bottleneck, Then Target It (`diagnose_then_target`) | Make Intermediate Steps Explicit and Score Them | 18 | 2 | 0 | 12 | 4 |
| `C03` | Dedicated Tokens Carrying Intermediate Evidence | Add a Dedicated Carrier for Intermediate State (`dedicated_state_carrier`) | Align Internal Representations to a Teacher | 26 | 1 | 0 | 18 | 7 |
| `C04` | Decompose Into Separately Rewarded Parts | Decompose Into Separately Verifiable Parts (`decompose_and_reward_parts`) | Substitute Cheap Verifiable Signals for Expensive Supervision | 15 | 0 | 2 | 10 | 3 |
| `C05` | Reward the Evidence Stage Separately | Make Intermediate Steps Explicit and Score Them (`reward_explicit_intermediates`) | Decompose Into Separately Verifiable Parts | 23 | 0 | 1 | 14 | 8 |
| `C06` | Symbolic-Proxy Data Synthesis | Synthesize Supervision From a Privileged Exact Representation (`privileged_proxy_synthesis`) | — | 49 | 2 | 0 | 28 | 19 |
| `C07` | Diagnose Failures, Then Target Training | Diagnose the Bottleneck, Then Target It (`diagnose_then_target`) | — | 44 | 1 | 2 | 28 | 13 |
| `C08` | Re-engineer Alignment for New Domain | Re-engineer the Interface, Keep the Backbone (`adapt_interface_not_backbone`) | Recast Everything Into One Shared Format | 38 | 0 | 3 | 22 | 13 |
| `C09` | Recast Inputs Into Native Instruction Format | Recast Everything Into One Shared Format (`unify_into_shared_format`) | Synthesize Supervision From a Privileged Exact Representation | 12 | 0 | 1 | 8 | 3 |
| `C10` | Cheap Verifiable Rewards Replace Rationales | Substitute Cheap Verifiable Signals for Expensive Supervision (`cheap_verifiable_proxy_reward`) | — | 41 | 0 | 3 | 29 | 9 |
| `C11` | Utility-Measured Training Data Selection | Measure Each Example's Utility and Select (`measure_example_utility`) | — | 12 | 0 | 0 | 6 | 6 |
| `C12` | Checkable Intermediate Artifacts as Rewards | Make Intermediate Steps Explicit and Score Them (`reward_explicit_intermediates`) | Substitute Cheap Verifiable Signals for Expensive Supervision | 85 | 2 | 2 | 58 | 23 |
| `C13` | Process-Level Reasoning Reward Shaping | Make Intermediate Steps Explicit and Score Them (`reward_explicit_intermediates`) | Repair a Degenerate Learning Signal | 15 | 0 | 1 | 10 | 4 |
| `C14` | Align Internals to Teacher Signals | Align Internal Representations to a Teacher (`distill_internal_property`) | Diagnose the Bottleneck, Then Target It | 25 | 2 | 2 | 13 | 8 |
| `C15` | Constrain Updates to Preserve Knowledge | Constrain Which Parameters May Change (`constrain_what_changes`) | — | 30 | 0 | 0 | 19 | 11 |
| `C16` | Restore Signal Under Sparse Rewards | Repair a Degenerate Learning Signal (`repair_degenerate_signal`) | — | 57 | 1 | 2 | 44 | 10 |
| `C17` | Self-Generated, Verified Experience Loops | Close a Self-Generated, Verified Experience Loop (`closed_self_verified_loop`) | Synthesize Supervision From a Privileged Exact Representation | 14 | 0 | 1 | 11 | 2 |
| `C18` | Shared Interface Bridging Planning and Control | Recast Everything Into One Shared Format (`unify_into_shared_format`) | Re-engineer the Interface, Keep the Backbone | 14 | 1 | 1 | 9 | 3 |
