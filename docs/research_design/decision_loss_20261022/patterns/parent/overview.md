# VLM post-training ideation patterns — overview

14 parent ideation patterns induced in one Opus 5.5 call from 19 sub-pattern clusters (RS §6). Counts come from per-paper multi-label tagging of all 879 papers (RS §6.5): `primary` = papers whose main move is this pattern. HC not assigned in this run. Full cards: `<id>.md`.

| id | name | primary clusters | primary n | oral | spotlight | poster | reject | supporting-only |
|---|---|---|---|---|---|---|---|---|
| `reward_explicit_intermediates` | Make Intermediate Steps Explicit and Score Them | C01 (Credit the Localization Subgoal), C05 (Reward the Evidence Stage Separately), C12 (Checkable Intermediate Artifacts as Rewards), C13 (Process-Level Reasoning Reward Shaping) | 123 | 3 | 4 | 81 | 35 | 47 |
| `diagnose_then_target` | Diagnose the Bottleneck, Then Target It | C02 (Isolate Weak Sub-Skill, Supervise Intermediates), C07 (Diagnose Failures, Then Target Training) | 94 | 2 | 4 | 64 | 24 | 76 |
| `cheap_verifiable_proxy_reward` | Substitute Cheap Verifiable Signals for Expensive Supervision | C10 (Cheap Verifiable Rewards Replace Rationales) | 89 | 1 | 4 | 55 | 29 | 74 |
| `privileged_proxy_synthesis` | Synthesize Supervision From a Privileged Exact Representation | C06 (Symbolic-Proxy Data Synthesis) | 86 | 1 | 3 | 52 | 30 | 35 |
| `manufacture_contrastive_pairs` | Manufacture Contrasts by Controlled Corruption | C00 (Label-Free Preference Pair Construction) | 71 | 2 | 0 | 56 | 13 | 45 |
| `closed_self_verified_loop` | Close a Self-Generated, Verified Experience Loop | C17 (Self-Generated, Verified Experience Loops) | 65 | 0 | 3 | 42 | 20 | 28 |
| `dedicated_state_carrier` | Add a Dedicated Carrier for Intermediate State | C03 (Dedicated Tokens Carrying Intermediate Evidence) | 61 | 0 | 1 | 46 | 14 | 19 |
| `adapt_interface_not_backbone` | Re-engineer the Interface, Keep the Backbone | C08 (Re-engineer Alignment for New Domain) | 58 | 2 | 5 | 36 | 15 | 29 |
| `repair_degenerate_signal` | Repair a Degenerate Learning Signal | C16 (Restore Signal Under Sparse Rewards) | 49 | 1 | 2 | 34 | 12 | 28 |
| `decompose_and_reward_parts` | Decompose Into Separately Verifiable Parts | C04 (Decompose Into Separately Rewarded Parts) | 44 | 0 | 2 | 28 | 14 | 40 |
| `measure_example_utility` | Measure Each Example's Utility and Select | C11 (Utility-Measured Training Data Selection) | 40 | 0 | 1 | 23 | 16 | 33 |
| `unify_into_shared_format` | Recast Everything Into One Shared Format | C09 (Recast Inputs Into Native Instruction Format), C18 (Shared Interface Bridging Planning and Control) | 39 | 0 | 5 | 25 | 9 | 14 |
| `distill_internal_property` | Align Internal Representations to a Teacher | C14 (Align Internals to Teacher Signals) | 31 | 1 | 1 | 20 | 9 | 15 |
| `constrain_what_changes` | Constrain Which Parameters May Change | C15 (Constrain Updates to Preserve Knowledge) | 29 | 0 | 1 | 17 | 11 | 11 |
