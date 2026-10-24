# Measure Each Example's Utility and Select
_id: `measure_example_utility`_

**Plain alias**. _Score every training example, keep the ones that pull weight_

**Definition**. Estimate the real contribution of individual training examples or sources via proxies, influence, uncertainty or meta-objectives, and select or reweight the pool accordingly.

**Operational signature**. question 'more data is better' → define a per-example utility estimate → score or learn weights cheaply → select/reweight → show subset matches or beats full set

**When to apply**. When the training pool is large, noisy or redundant and compute or quality suffers from using it uniformly.

**Evidence base**. primary-tagged n=40, oral 0, spotlight 1, poster 23, reject 16; supporting-only n=33. The success side is a single spotlight, and there are no orals (the corpus has only 13 in total), so posters carry most of the accepted-side evidence. One poster (WWo9G5zyh0) has no public reviews and is excluded from the reviewer-expectation evidence.

## Success conditions (from Oral/Spotlight; Poster as support)
- **The selected subset beats random selection consistently at a matched budget, across backbones and benchmarks** — The spotlight shows strong results on eight benchmarks at two model scales, with explicit ablations against random and online selection. One poster is praised as "the first method to consistently beat random selection" across 10 benchmarks and several target models. Other posters beat random subsampling at matched budgets on two backbones, or show a filtered top 30% beating both the full data and similarity filtering. Random selection is a strong baseline in this family, so clearing it decisively is the minimum evidence that the utility estimate measures something real. [ids: PHu9xJeAum, ZptT82VZ4E, NWRoNoJ9QX, KIximJjqjV, pXXKKBx8G0]
- **A diagnostic finding or a stated mechanism comes before the selector** — Accepted papers first show why uniform use fails, then build the score from that. Examples: saturation explained by gradient variance over label mixture and reliability; non-monotonic harm from high-uncertainty samples; imbalanced reward growth on easy versus hard pairs; attention collapse under poisoning; an exploratory finding that higher compositional complexity helps. The diagnosis turns a heuristic into a justified measurement. [ids: NWRoNoJ9QX, LuZjiUNuFL, zvZTIXkzDe, os4QYDf3Ms, 073WQjmWKU]
- **The utility signal is specific to the learner and nearly free, read from quantities the model or the data already produce** — Examples: search iterations needed by the current model (spotlight), Monte Carlo statistics already in the annotations, the model's own with-versus-without-input logit gap, token confidence, and pretrained likelihood of a target. Cheapness preserves the efficiency claim, and specificity to the learner explains why the subset helps this model. [ids: PHu9xJeAum, NWRoNoJ9QX, NsxthTVpqA, HK0IhFBmoN, iyO15Uqy75]
- **Cost accounting, fair baselines and out-of-distribution checks are added when challenged** — The spotlight rose above poster only after rebuttal added fair baselines, cost figures and out-of-distribution evidence. Posters likewise moved after adding six data-reduction baselines and an open-source generator, overhead comparisons, or cost and out-of-distribution reports. [ids: PHu9xJeAum, 073WQjmWKU, T4LtGj7us1, eZu358JOOR, EwFJaXVePU]
- **The scorer or the selected subset is reused beyond one training run** — A trained quality scorer also serves as a reward and a best-of-N reranker. A reward model is reused to filter a noisy downstream training set. A small proxy model selects data for a larger target. Reuse shows that the utility estimate generalizes rather than overfitting one setup. Evidence here is poster-only. [ids: KIximJjqjV, eZu358JOOR, ZptT82VZ4E]

## Failure modes (from Reject)
- **Selection cost is unaccounted for, or approaches the cost of training** — Examples: gradient-influence scoring with a warm-up reference model, influence plus gradient trajectories, attention extraction over the whole pool, and about 6x compute for test-time selection. Reviewers judged that the efficiency premise collapsed. [ids: BydkbNH0gj, 7LG9YBnadZ, HMkwfZl61l, ckfNLA1Dxy, ZpXi8cF5b0, eZb7N6fc5n]
- **Gains over random, uniform or simple baselines are marginal** — Examples: 1–2% over random sampling, about 0.9% average, small gains over uniform or expert mixture weights, and 1–3 points over the strongest baselines. Gains this small cannot justify a multi-stage selector. [ids: zO8LrT9Sy9, yB09CcjoII, vk16mWMGXz, 7zm2yY5BEI, rLySPkyl3S]
- **Established selection baselines are missing, so the paper compares only against random selection or its own variants** — Influence-based, meta-weighting, instruction-mining and gradient-selection methods were absent. [ids: DNvzCsQG1D, HMkwfZl61l, ckfNLA1Dxy, YyCgWQFGtL, ZpXi8cF5b0]
- **One model, one dataset or one small scale, with gains shrinking on larger models** — Generality of the utility estimate is the claim, so a single setting cannot carry it. [ids: DNvzCsQG1D, 7LG9YBnadZ, eZb7N6fc5n, zO8LrT9Sy9, yB09CcjoII, YyCgWQFGtL]
- **The utility construct is unjustified, heuristic, or proved only in a toy setting** — Examples: "intrinsic capabilities" that are subjective and sensitive to clustering, theory limited to a single-layer single-head model, an alignment objective with no stated link to generalization, unjustified layer choices, and "less is better" claims confounded by uncontrolled training steps. [ids: 7LG9YBnadZ, HMkwfZl61l, vk16mWMGXz, ckfNLA1Dxy, zO8LrT9Sy9]
- **Recombination of known reweighting or filtering components presented as the contribution** — Examples: perplexity filtering plus model soups, expert-scored filtering in an engineering recipe, an instance-level extension of an accepted domain-level reweighting method, and a curriculum plus length reward. [ids: gpP68EP1Jn, SboD1FBYdy, xmP7ZpjPQ5, 7zm2yY5BEI, BydkbNH0gj]

## Oral vs Reject gap
confidence: low

The success side has a single spotlight, so this comparison leans on posters as accepted support. The spotlight (PHu9xJeAum) differs from the rejects on three observable points:

1. Its utility signal is specific to the learner and directly tied to the training objective.
2. The data reduction is an order of magnitude, paired with a new best benchmark result, rather than a 1–3 point gain.
3. Its rebuttal supplied fair baselines, cost figures and out-of-distribution evidence, which are exactly what rejects lacked.

Two paired cases confirm the direction:

- **Same method, two venues.** The attention-trajectory selector was rejected (HMkwfZl61l: toy-model theory, incremental over activation clustering, no cost accounting, missing gradient-selection baselines). Resubmitted with evaluation across 10 benchmarks and several target models, it was accepted as a poster (ZptT82VZ4E: praised as the first to consistently beat random). The theory–practice and overhead concerns remained, so breadth and baseline coverage, not new theory, moved the decision.
- **Accepted method versus its extension.** The domain-level bi-level reweighting paper was accepted (ZyiBk1ZinG), while its instance-level extension, submitted without a rebuttal, was rejected as incremental (xmP7ZpjPQ5).

With no orals (n_oral = 0 < 3), the oral-only version of this gap cannot be assessed.

## Oral vs HC gap
Not assessed: HC labels were not assigned in this run (citation lookup incomplete); the reviewer-expectation panel carries the contrast.

## Reviewer expectations
In this section, "oral reviews" means the reviews of the single spotlight; no orals exist in this parent.
- **Account for selection cost against the training savings it claims** _(source: both)_ [ids: PHu9xJeAum, ZptT82VZ4E, BydkbNH0gj, HMkwfZl61l, ckfNLA1Dxy, ZpXi8cF5b0]
- **Compare against established data-selection, curriculum or active-learning methods, not only random selection or internal heuristics** _(source: both)_ — Raised against the spotlight ("baseline comparisons unclear or unfair"), against posters, and against rejects. [ids: PHu9xJeAum, NWRoNoJ9QX, T4LtGj7us1, LuZjiUNuFL, DNvzCsQG1D, HMkwfZl61l]
- **Show the result across several backbones, datasets and model scales** _(source: both)_ — Praised in the spotlight (two scales, eight benchmarks) and cited as missing in rejects and borderline posters. [ids: PHu9xJeAum, DNvzCsQG1D, 7LG9YBnadZ, pXXKKBx8G0, zvZTIXkzDe]
- **Explain why the selected subset wins, with a mechanism rather than correlation** _(source: reject reviews)_ — Also raised in poster reviews. [ids: zO8LrT9Sy9, BydkbNH0gj, 073WQjmWKU, S4HPn5Bo6k, LYBcBH3Y76]
- **State how reusable the selection is across models; a utility that is specific to the learner limits the value of the released subset** _(source: both)_ [ids: PHu9xJeAum, ZptT82VZ4E, 7LG9YBnadZ, ckfNLA1Dxy]
- **Report significance, run multiple seeds, and check sensitivity to thresholds and hyperparameters** _(source: both)_ [ids: PHu9xJeAum, DNvzCsQG1D, LuZjiUNuFL, yB09CcjoII, NWRoNoJ9QX, xmP7ZpjPQ5]
- **Rule out overlap between the data used to compute utility or rewards and the evaluation sets** _(source: reject reviews)_ — Also raised repeatedly in poster reviews. [ids: 7zm2yY5BEI, T4LtGj7us1, ZyiBk1ZinG, ULJ4gJJYFp, eZu358JOOR]

## Cognitive barriers
- "More data is better" and "every annotated example is equally useful" are default assumptions, so redundancy and harmful examples go unnoticed until someone tests reduced subsets (zO8LrT9Sy9, PHu9xJeAum, NWRoNoJ9QX, LuZjiUNuFL, T4LtGj7us1).
- Data quality is treated as something defined by hand-made heuristics, not as a quantity that can be learned from downstream validation feedback (DNvzCsQG1D, ZyiBk1ZinG, xmP7ZpjPQ5).
- The model's own internal statistics (logit contrasts, attention entropy, alignment views, token confidence) are seen as objects of analysis or inference-time tricks, not as free training-time utility signals (NsxthTVpqA, os4QYDf3Ms, LYBcBH3Y76, HK0IhFBmoN, ckfNLA1Dxy).
- A single global score or hyperparameter is assumed to fit all data, which hides the fact that utility varies by task, skill group or pair difficulty (EwFJaXVePU, zvZTIXkzDe, yB09CcjoII, BydkbNH0gj).
- The right notion of utility (gradient similarity, per-step label correctness) is either intractable or silently biased by the tool used to estimate it. The barrier is finding a cheap proxy that provably bounds it, or recognizing the estimator as a source of bias (ZptT82VZ4E, HMkwfZl61l, YyCgWQFGtL).

## Representative examples
| forum_id | label | one-line what it does |
|---|---|---|
| PHu9xJeAum | spotlight | Counts the tree-search iterations each sample needs, keeps hard-but-solvable items, and runs RL on a tiny subset |
| ZptT82VZ4E | poster | Cross-modal attention-spectrum trajectories from a small proxy stand in for gradient similarity; consistently beats random |
| NWRoNoJ9QX | poster | Gradient-variance analysis gives a zero-cost score from existing labels; about 10% of the data matches full-data training |
| LuZjiUNuFL | poster | Finds that high-annotator-uncertainty samples hurt; selects a mid-uncertainty window and self-labels from a small seed |
| HMkwfZl61l | reject | Earlier submission of the attention-trajectory selector; rejected for toy theory, cost and missing baselines |
| BydkbNH0gj | reject | Task-difficulty budget plus gradient influence; scoring cost rivals training and gains are marginal |
| zO8LrT9Sy9 | reject | Clustering coreset for a perceptual task; only 1–2% over random, with the scaling claim confounded |
| DNvzCsQG1D | reject | Learns a selector from cluster-level validation loss to pick 200 examples; one model, weak baselines |
