# Decompose Into Separately Verifiable Parts
_id: `decompose_and_reward_parts`_

**Plain alias**. _Split the task, reward each piece_
**Definition**. Break a compositional or multi-capability task into sub-tasks, branches or modules, each with its own verifiable or cross-part reward, and optimize them jointly under one objective.
**Operational signature**. factor task into parts → give each part a verifiable reward → couple parts via cross-part or consistency terms → optimize jointly (optionally staged to avoid conflict)
**When to apply**. When a task is a composition of separable capabilities whose individual correctness is cheap to verify.
**Evidence base**. primary-tagged n=44, oral 0, spotlight 2, poster 28, reject 14; supporting-only n=40

Note on evidence: no primary paper is an oral (the corpus has only 13 orals in total). The success side is the 2 spotlights, so posters serve throughout as supporting accepted evidence. Two primary-tagged pairs are the same work rejected at one venue and accepted as a poster at the next (88sop0LAIg → 99z6N9kXR6; 4Sv5MQ931E → dtmTQawQN2). They give an unusually clean view of what changed.

## Success conditions (from Oral/Spotlight; Poster as support)
- **The decomposition answers a diagnosed failure of the monolithic or scalar-reward baseline** — accepted papers first show what the undivided approach gets wrong: imitation of interleaved traces scoring below the text-only baseline, a pilot study where joint training beats separate training, advantages from different reward terms cancelling, correct answers unsupported by the gathered evidence, easy sub-categories dominating a summed reward. The split is then the targeted fix rather than an architectural preference. [ids: JT4zAf3zDs (spotlight), aDa0xEFDu1 (spotlight), cRJNk4bhZi, RsPequeaiZ, 6dcgmtBYhh]
- **Each part's reward checks its functional role in producing the outcome, not just its format** — the strongest designs reward whether an intermediate artifact is correct, usable and actually consumed downstream, or feed a later stage's quality back as the reward for an earlier stage (refinement quality rewarding the coarse proposal; answerer loss ranking candidate plans; in-the-loop reward from the consumer). [ids: JT4zAf3zDs (spotlight), unMwI4JLrP, dtmTQawQN2, vlraTIgUD3, elrnTLDiT3]
- **A cross-part coupling term is shown to beat optimizing the parts separately** — a cycle-consistency or cross-feedback reward between parts, with evidence that joint optimization outperforms separate or alternating training, is what makes the result read as synergy rather than a pipeline. [ids: aDa0xEFDu1 (spotlight), unMwI4JLrP, elrnTLDiT3]
- **Per-part signals are kept separate so they cannot cancel or dominate** — separate advantage normalization per reward dimension or content category, separate preference weights per response segment, or disentangled reward and cost signals under a constraint. Poster-only evidence. [ids: cRJNk4bhZi, 6dcgmtBYhh, L9vV3wVC72, OIH3T5ZPBW, KwWYvt547M]
- **Ablations show every part's reward is necessary, and the verifier is checked against humans** — reviewers credit papers that show removing any one verifier or reward hurts, and that the composite verifier correlates with human judgment. [ids: JT4zAf3zDs (spotlight), aDa0xEFDu1 (spotlight), DG0F1cdjN7, iRVbPxHNrX]
- **The decomposition unlocks cheaper supervision or doubles as measurement** — a coarse interface extractable from action-free data, a few thousand templated samples instead of large caption corpora, recombinable primitive factors with free sub-answers, or the same atomic-unit check serving as both metric and reward. [ids: h7aQxzKbq6, DG0F1cdjN7, FnF3UjiN11, 636M0nNbPs, iRVbPxHNrX, JT4zAf3zDs (spotlight)]

## Failure modes (from Reject)
- **Standard policy optimization plus a task-specific composite reward, presented as the method** — reviewers repeatedly called these incremental recipes: shaping terms on region proposals, matched-pair rewards for one captioning task, surrogate structure-recovery rewards, tool-orchestration rewards, rule checks bolted onto a generator. [ids: 1QaTq2NXD6, 88sop0LAIg, bQTlwMXXwW, QHaV1ZHzi7, SaAzsJQECU, RDz3EPC3Lp, nRhHbKP1y9, 1UqnezioFf]
- **The part rewards are hand-crafted, privileged or judged by an unvalidated model** — stage boundaries from privileged geometric event detectors, a heuristic token-sensitivity score, oracle-model rationales used as process references, and uncalibrated language-model judges for both reward and evaluation. [ids: TYGtbbccnm, Q4KIDjDDRJ, QHaV1ZHzi7, kC4JPVo62f, 1QaTq2NXD6]
- **What each part contributes is never isolated** — no comparison of joint versus isolated training, missing per-reward or weight ablations, stage semantics not separated from plain reward densification, algorithm gains tangled with synthetic data scale, or a motivating ablation that does not show the claimed interdependence. [ids: HfRJDTBuSr, kC4JPVo62f, TYGtbbccnm, QHaV1ZHzi7, 1UqnezioFf, Q4KIDjDDRJ]
- **Evaluation does not exercise the composed system in its real setting** — offline single-step accuracy with no closed-loop success, train and evaluation from the same source, a self-built synthetic benchmark only, outdated or saturated datasets, a single domain, or a 30-item test set. [ids: HfRJDTBuSr, kC4JPVo62f, nRhHbKP1y9, 4Sv5MQ931E, 88sop0LAIg, SaAzsJQECU]
- **The modular pipeline's overhead is not repaid** — about 6x slower inference for modest gains, unreported latency from tool scheduling and judges, and stacks of opaque reward models that hurt reproducibility. [ids: 4Sv5MQ931E, SaAzsJQECU, RDz3EPC3Lp, Te3FFNlk21]
- **The headline claim outruns what the decomposition delivers** — "resolving" a trade-off while one objective drops sharply, a static single-step instantiation framed as a general capability, and explainability or causal-reasoning claims left unvalidated. [ids: Te3FFNlk21, 1QaTq2NXD6, QHaV1ZHzi7, 88sop0LAIg]

## Oral vs Reject gap
With the success side limited to the two spotlights, the observable gap is this. Both spotlights start from a surprising, measured failure of the simpler regime (imitation hurting below the text-only baseline; separate training losing to joint training) and design part rewards around the functional or causal role of the intermediate step (the generated artifact must mediate the answer; generation and understanding must check each other through cycle consistency). They then ablate each reward. The 14 rejects mostly decompose a task along an intuitive pipeline (plan/execute, propose/select, perceive/reason, stage-by-stage), attach heuristic or judge-scored rewards to each piece, and leave both the necessity of each piece and the benefit of coupling untested. They also often evaluate offline, synthetically or on saturated data. The two resubmitted pairs support the evaluation part of this gap: the same method moved from reject to poster after broader multi-domain benchmarks and fuller ablations, while the novelty concern stayed. This suggests evaluation breadth decides the reject/poster line, and the diagnostic and functional-reward moves decide the higher tier. Because n_oral = 0 < 3, the oral-only version of this gap cannot be assessed. confidence: low

## Oral vs HC gap
Not assessed: HC labels were not assigned in this run (citation lookup incomplete); the reviewer-expectation panel carries the contrast.

## Reviewer expectations
Here "oral reviews" means the success-side reviews, which for this parent are the 2 spotlights (no orals are primary-tagged).
- **Show that coupled or joint optimization beats training the parts separately or in alternation** _(source: both)_ [ids: aDa0xEFDu1, HfRJDTBuSr, kC4JPVo62f]
- **Ablate each part's reward and justify its weight** _(source: both)_ — the spotlights were praised for showing every verifier or reward is necessary; rejects were faulted for missing per-reward ablations or weights all set to 1. [ids: JT4zAf3zDs, aDa0xEFDu1, QHaV1ZHzi7, kC4JPVo62f, bQTlwMXXwW]
- **Validate the part verifiers and judges, and address reward hacking** _(source: both)_ [ids: JT4zAf3zDs, aDa0xEFDu1, kC4JPVo62f, QHaV1ZHzi7]
- **Evaluate on established external benchmarks and more than one model scale or family** _(source: both)_ [ids: JT4zAf3zDs, aDa0xEFDu1, nRhHbKP1y9, HfRJDTBuSr, 88sop0LAIg]
- **Keep the stated claim consistent with the design and results** _(source: both)_ — a task-specific second stage that contradicts a unified-optimization claim; a trade-off "removed" while one metric drops; an over-scoped framing. [ids: aDa0xEFDu1, Te3FFNlk21, 1QaTq2NXD6]
- **Show novelty beyond known decomposition plus verifiable-reward ingredients** _(source: both)_ [ids: JT4zAf3zDs, 1QaTq2NXD6, bQTlwMXXwW, SaAzsJQECU, 88sop0LAIg]
- **Motivate the split with a clear diagnostic finding about the undivided approach** _(source: oral reviews)_ [ids: JT4zAf3zDs, aDa0xEFDu1]
- **Report the latency and compute of the multi-part pipeline against its gains** _(source: reject reviews)_ [ids: 4Sv5MQ931E, SaAzsJQECU, 88sop0LAIg, RDz3EPC3Lp]

## Cognitive barriers
- The end-to-end default assumes one large model must produce every output, so a coarse intermediate interface, a cheap first-stage filter, or a decoupled specialist looks like a step backward rather than a lever. [h7aQxzKbq6, 88sop0LAIg, SaAzsJQECU, whRAOJiyHM]
- Summing several reward terms or metrics into one scalar looks harmless. That cancellation across dimensions, or domination by easy sub-parts, only shows up if one inspects per-dimension advantages. [cRJNk4bhZi, 6dcgmtBYhh, L9vV3wVC72, OIH3T5ZPBW]
- Intermediate parts have no ground truth, which seems to block per-part supervision. The way out is to use the downstream part's outcome or loss as the earlier part's reward. [dtmTQawQN2, 4Sv5MQ931E, unMwI4JLrP]
- Imitation and data scaling are the defaults, so a drop from more structured data is dismissed rather than read as missing functional dependence. Accuracy-only scoring similarly hides correct-but-unsupported answers. [JT4zAf3zDs, lUHwUuQGdV, RsPequeaiZ]
- The target is treated as an atomic label or holistic judgment. Seeing it as a composition of cheaply checkable sub-skills or units is a reframing step. [DG0F1cdjN7, NBOHB6aYZh, FnF3UjiN11, 636M0nNbPs, QHaV1ZHzi7]

## Representative examples
| forum_id | label | one-line what it does |
|---|---|---|
| JT4zAf3zDs | spotlight | Shows imitation of tool-interleaved traces copies format only, then uses RL with three verifiers on the generated artifact's correctness, usability and intent |
| aDa0xEFDu1 | spotlight | Jointly reinforces understanding and generation in one model with cross-task cycle-consistency and self-scored matching rewards, then specializes per task |
| cRJNk4bhZi | poster | Computes advantages separately per reward dimension and groups rollouts by capability type to stop reward signals cancelling |
| unMwI4JLrP | poster | Coarse-to-fine localization where the refinement stage's quality is fed back as a delayed reward for the coarse proposal stage |
| 636M0nNbPs | poster | Splits long outputs into atomic verifiable units and reuses the same unit check as both evaluation metric and preference reward |
| 99z6N9kXR6 | poster | Cheap candidate proposal plus an RL-trained verifier rewarded for rejecting bad candidates; accepted after broader multi-domain evaluation |
| 88sop0LAIg | reject | Earlier version of the same candidate-then-reason method; rejected as an incremental combination evaluated on one domain with outdated baselines |
| TYGtbbccnm | reject | Stage-wise costs and shaped rewards from hand-built privileged event detectors; stage semantics not isolated from plain reward densification |
