# Re-engineer the Interface, Keep the Backbone
_id: `adapt_interface_not_backbone`_

**Plain alias**. _Fix how the new input gets in, not the model it goes into_
**Definition**. Extend a pretrained system to a new input type or regime by redesigning how that input is lifted, tokenized, routed, compressed or aligned into the existing interface, rather than retraining from scratch.
**Operational signature**. diagnose mismatch between new input and existing interface → design lifting/routing/compression bridge → align on paired data → fine-tune with minimal backbone change
**When to apply**. When a strong pretrained system exists but a new input structure does not fit its native interface.
**Evidence base**. primary-tagged n=58, oral 2, spotlight 5, poster 36, reject 15; supporting-only n=29

Note on the evidence: the success side has 7 papers (2 orals, 5 spotlights). The corpus has only 13 orals in total, so the two here (DE5ZJtR4bg, oJ84bedrtM) are a meaningful but small signal. Posters are cited as supporting accepted evidence.

## Success conditions (from Oral/Spotlight; Poster as support)
- **The mismatch is identified precisely, with evidence, before the bridge is designed** — Both orals start by showing what the existing interface cannot carry. One gives an analytic argument that an input-only model cannot separate object size and depth from an omitted acquisition parameter. The other runs a preliminary study showing a shared adapter is dominated by one input type. The fix is then aimed exactly at that gap. Posters praised for motivation do the same: they locate the failure in a language-biased input representation, or in an over-smoothed final-layer readout. [ids: DE5ZJtR4bg, oJ84bedrtM, Yx0bl5OFvc, WKTNdU155n, 0fib2BYc0L]
- **The backbone keeps what it is good at, and only the misfitting role is moved out** — Spotlights win by assigning each role to the right place. Per-step comprehension stays in the pretrained network while temporal history moves to a light policy head, avoiding costly co-fine-tuning. Features from a well-pretrained neighboring input type are lifted into the new structure instead of training a native encoder. A non-native input is rendered into the model's native input space. [ids: lFYj0oibGR, YQA28p7qNz, 5fm6XlWpiY, 1tZbq88f27]
- **The interface is general enough to cover the input's variants, so one bridge serves many tasks or formats** — A hybrid discrete-plus-pooled-feature encoding handles points, boxes, scribbles and free-form masks for both input and output. A format-agnostic converter transfers across structural layouts. A budget-preserving scheme covers both short and long inputs in one model. Reviewers credit this unification as the conceptual contribution. [ids: 2msbbX3ydD, 5fm6XlWpiY, FS3FzdrFZ7, PN7l7mBPDO]
- **Data efficiency or out-of-distribution transfer is demonstrated, not just in-distribution gains** — Successes show that the bridge, not scale, does the work: about 10K alignment examples that transfer to unseen layouts, a randomized-acquisition test across setups, zero-shot generalization with much less data than the prevailing recipe, or ~1% of competitors' data in the new regime with no regression in the original one. [ids: PN7l7mBPDO, DE5ZJtR4bg, lFYj0oibGR, Yx0bl5OFvc]
- **New paired data or a benchmark is shipped alongside the bridge** — Several spotlights pair the interface with the large paired corpus that alignment needs (refer-and-ground instructions with mined hard negatives, generated structure-language pairs, stepwise planning data from first-person recordings). Reviewers list these as primary contributions. [ids: 2msbbX3ydD, YQA28p7qNz, IL5zJqfxAa, PRCizVyL1K]
- **Gains are consistent across backbones and settings** — The adapter-redesign oral holds across three mixed-input settings and several backbones. Accepted posters likewise show effects across model scales or encoder families rather than one pairing. [ids: oJ84bedrtM, PN7l7mBPDO, 0fib2BYc0L, iCvueZ8KaN]

## Failure modes (from Reject)
- **The paper ports a recipe or adds a channel and calls that the contribution** — Rejects often take the standard encode-project-instruction-tune recipe to a new signal or domain, or add the missing input channel to an existing pipeline, using a standard encoder and the prior training recipe. Reviewers call the step "natural" and the novelty "very limited". [ids: vFfVXSP24J, 1WZbF2G60R, DzxaRFVsgC, oCIEUHJjNj, 69IxzbIiVb]
- **The bridge cannot be told apart from an existing adapter, query or prompt mechanism** — Learnable probe queries read as a Q-Former variant. Input-level tensors read as prompt tuning or a guard model. Peripheral adapters on a frozen server model repeat prior split designs. Retrieval plus FiLM plus a weighted loss reads as known parts. [ids: WJw612OZhD, crOQJrWrdU, YsruHaTNYS, tVN0E4LnJV, xPQnTGh9zN]
- **The hypothesized mismatch is not substantiated** — The foundational causal claim rests on one experiment with sub-percent changes, is not separated from known attention-dilution or modality-imbalance effects, or is supported only by qualitative visualizations that leave "added capacity" as the simpler explanation. [ids: dT2gGWoO6m, xPQnTGh9zN, WJw612OZhD, lU4LPHfvaF]
- **The representation's contribution is confounded with backbone or data** — Headline comparisons use a stronger backbone or much more training data than baselines, with no controlled ablation that isolates the new interface. [ids: 11VdnqNsbm, DzxaRFVsgC, 69IxzbIiVb]
- **The pipeline is stitched from many components without design justification** — Several auxiliary objectives, modified attention and extra training phases are presented together. Reviewers read them as tricks, see incremental gains that do not justify the added parameters, and find ablations of the many loss terms missing. [ids: lU4LPHfvaF, uxYbEAEWm4, xPQnTGh9zN]
- **Evaluation is narrow and the paper is not defended** — One base model, one domain, multiple-choice evaluation on the training datasets, or a tiny dataset without variance. Four rejects posted no rebuttal at all, so these concerns stood. [ids: Yc7PYzWWV8, 1WZbF2G60R, YsruHaTNYS, tVN0E4LnJV, vFfVXSP24J]
- **The bridge's inference cost is not accounted for** — Repeated calls to two models, an added gating or probe pathway, or token selection with no efficiency analysis. [ids: tVN0E4LnJV, dT2gGWoO6m, WJw612OZhD]

## Oral vs Reject gap
The success side (oral ∪ spotlight) treats the interface mismatch as the finding and the bridge as its consequence. It names what the native interface cannot express (an omitted acquisition variable, one input type dominating a shared adapter, region shapes that coordinates cannot encode, history that should not live in the backbone). It then puts a minimal fix exactly there, and shows the fix works because of that diagnosis: cross-setup transfer, data efficiency, consistency across backbones. Rejects mostly invert this order. They start from an available mechanism (a probe pathway, an input-level tensor, a gating module, a borrowed recipe) and assert a mismatch afterwards. The asserted mismatch is either weakly evidenced or indistinguishable from a known one, and the comparison does not isolate the interface from backbone or data changes. Datasets appear on both sides, so a dataset alone does not separate accept from reject. What separates them is whether the paired data serves a precisely diagnosed interface gap. The oral-only version cannot be assessed (n_oral = 2).

## Oral vs HC gap
Not assessed: HC labels were not assigned in this run (citation lookup incomplete); the reviewer-expectation panel carries the contrast.

## Reviewer expectations
In this parent, "oral reviews" means success-side reviews (oral ∪ spotlight).
- **Run a controlled ablation that isolates the interface from backbone and training-data changes, with baselines trained on the same data** _(source: both)_ [ids: YQA28p7qNz, 11VdnqNsbm, DzxaRFVsgC, gVbPWbA97s, 5fm6XlWpiY]
- **Give direct evidence for the diagnosed mismatch (analytic argument, preliminary study, attention or gradient analysis) rather than an intuitive story** _(source: both)_ [ids: oJ84bedrtM, DE5ZJtR4bg, dT2gGWoO6m, y13mtWvmoG, V2yikPhdXE]
- **Compare against the simplest alternative bridge, such as raising adapter rank, a cascade plus metadata, plain prompt tuning, a regression head, or a side-by-side with the prevailing adapter** _(source: both)_ [ids: oJ84bedrtM, lFYj0oibGR, crOQJrWrdU, K4FAFNRpko, nKJGjovmZz, BqHaLnans2]
- **State clearly how the work differs from the closest existing interface mechanism** _(source: both)_ [ids: 2msbbX3ydD, WJw612OZhD, crOQJrWrdU, YsruHaTNYS, rbIlWjTFKj]
- **Show the bridge generalizes beyond one benchmark, domain or simulated variation** _(source: both)_ [ids: lFYj0oibGR, DE5ZJtR4bg, Yc7PYzWWV8, 11VdnqNsbm]
- **Report the latency, parameter and compute overhead the bridge adds** _(source: both)_ [ids: oJ84bedrtM, PN7l7mBPDO, tVN0E4LnJV, oUYmk8WaG0, FOkKndty5B]
- **Validate generated paired data and rule out leakage into evaluation** _(source: both)_ [ids: YQA28p7qNz, oCIEUHJjNj, QKOdOlfzSe, Av10FtirO4]
- **Check that the backbone's original capabilities are preserved after adaptation** _(source: both)_ [ids: lFYj0oibGR, Yc7PYzWWV8, Yx0bl5OFvc, t877958UGZ, BqHaLnans2]

## Cognitive barriers
- Strong aggregate or in-distribution scores hide the mismatch. End-to-end training learns setup-specific shortcuts, and a shared adapter that ignores one input type still posts fine task numbers, so the interface flaw stays invisible until a cross-setup or per-input-type probe exposes it (DE5ZJtR4bg, oJ84bedrtM, K4FAFNRpko).
- A new input type is assumed to need a new native encoder, joint training or explicit structure. That the existing backbone already holds the capability, and only needs input lifted, re-projected or rendered into its native space, runs against the default (1tZbq88f27, YQA28p7qNz, 0fib2BYc0L, rbIlWjTFKj, 5fm6XlWpiY).
- The prevailing recipe asks the backbone to do every role. It emits actions, timestamps, coordinates and response timing as tokens, and it is hard to see that one of those roles should move to a dedicated head, pointer or check (lFYj0oibGR, 4n7IifN7yr, d6vMek58Zv, 5fSkinHw7w).
- A recipe copied unchanged from one input type misses what is specific to the new one: over-smoothed final layers in graphs, language-biased input tokens, adapters borrowed from single-input models (WKTNdU155n, oJ84bedrtM, Yx0bl5OFvc).
- Adaptation is framed as a binary choice: freeze (limited specialization) or fine-tune (forgetting). Third options, such as letting the backbone read new tokens natively, adding a parallel pathway, or a placeholder dropped at inference, are not considered (BqHaLnans2, WJw612OZhD, t877958UGZ, Yc7PYzWWV8).

## Representative examples
| forum_id | label | one-line what it does |
|---|---|---|
| DE5ZJtR4bg | oral | Proves input-only models cannot separate size and depth from an omitted acquisition parameter, then conditions every token on it, varies it in training and distills priors |
| oJ84bedrtM | oral | Shows a shared low-rank adapter is dominated by text tokens and redesigns it with per-input-type down-projections, low-rank cross-attention and a shared up-projection |
| lFYj0oibGR | spotlight | Keeps per-step comprehension in the pretrained model and moves temporal history to a lightweight policy head, tuning only connectors on a small imitation set |
| 2msbbX3ydD | spotlight | Encodes any-shape region as text coordinates plus mask-pooled features so one interface serves both referring and grounding, with a large instruction set |
| PN7l7mBPDO | spotlight | Aligns structure and content separately on ~10K examples, then reasons global-to-local over an extracted sub-region with no external parser |
| dT2gGWoO6m | reject | Information-bottleneck token gating and cluster pruning, built on a redundancy-causes-hallucination hypothesis supported by sub-percent evidence |
| WJw612OZhD | reject | Learnable probe queries cross-attending to intermediate frozen-encoder layers for domain shift; read as a Q-Former variant with only qualitative evidence |
| 1WZbF2G60R | reject | Adds the missing perceptual channel to an existing symbolic-only pipeline and retrains with the prior recipe; small data, no rebuttal |
