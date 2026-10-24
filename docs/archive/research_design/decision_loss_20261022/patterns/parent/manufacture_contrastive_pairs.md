# Manufacture Contrasts by Controlled Corruption
_id: `manufacture_contrastive_pairs`_

**Plain alias**. _Break it on purpose to get negatives_
**Definition**. Create preferred/rejected training pairs without human labels by deliberately degrading inputs, prompts or outputs (or reading the model's own error signals), then concentrate the learning signal on where the pair differs.
**Operational signature**. pick a known failure mode → synthesize corrupted counterpart → pair with clean output → localize the difference → train contrastively on it
**When to apply**. When preference or negative data is expensive but failure modes are known well enough to induce them automatically.
**Evidence base**. primary-tagged n=71, oral 2, spotlight 0, poster 56, reject 13; supporting-only n=45

Note on evidence: the success side (oral ∪ spotlight) is only the 2 orals; the corpus has 13 orals in total. Success conditions below therefore lean on posters as supporting accepted evidence, and each condition says which ids are oral.

## Success conditions (from Oral/Spotlight; Poster as support)
- **The corruption is derived from a measured failure, not an assumed one** — accepted papers first show where the model actually fails (a hand-annotated benchmark exposing weak verification; hallucinated evidence hiding inside correct-looking rationales; a manual error audit finding most errors are perception errors; statistical factors behind fabricated content), and only then synthesize negatives that reproduce that failure. The diagnosis makes the negative read as necessary rather than arbitrary. [ids: DM0Y0oL33T (oral), ZymCPON45y (oral), izbBqTL8vb, oZDJKTlOUe, 7lpDn2MhM2, ekOwxTn65Y]
- **The pair differs in a minimal, known place, and the loss is concentrated there** — near-duplicate inputs with matched queries, single-attribute edits, phrase swaps or perturbed spans give pairs where the difference is known by construction. The objective then puts weight on the differing span or element instead of the whole sequence, which reviewers credit as precise credit assignment and less drift from the base model. [ids: ZymCPON45y (oral), yG1fW8igzP, Zy2XgaGpDw, fgfftCWPrn, fJRuMulPkc, tbqxbGMFUV]
- **The contrast is tied to the evidence input, so the shortcut cannot win** — the corrupted side removes, masks, reverses or shuffles the evidence (or contrasts two nearly identical inputs) so that the preferred output is only preferable *given* the real evidence. This blocks the prior-only solution that output-side text edits leave open. [ids: ZymCPON45y (oral), ekOwxTn65Y, izbBqTL8vb, BfWFCVjsNe, OYciB30Z4n, a2JTVVvcEl, TcICHju9Mo]
- **The manufactured pairs train a transferable skill or drive a new procedure, not just one extra loss term** — the strongest papers decompose the target into a few atomic skills that transfer across tasks, or reuse the model's own contrast ability as a verifier inside a reason-contrast-rethink or verify-and-edit loop. Posters show a weaker version of this: one generator serving as both benchmark and training data, or a per-element detector reused for several downstream jobs. [ids: DM0Y0oL33T (oral), ZymCPON45y (oral), bJddXCyosA, Zy2XgaGpDw, 02NbD16OnA, QGv6QwDA4z]
- **Pair-design choices are analysed, not just used** — accepted papers report which pair types, difficulty levels, ratios or masking strategies work and which skills do not generalize. Reviewers treat this as actionable guidance and as evidence that the effect comes from the contrast, not the data volume. [ids: ZymCPON45y (oral), DM0Y0oL33T (oral), izbBqTL8vb, J44HfH4JCg, fJRuMulPkc]
- **The objective stays consistent when the two sides of the pair have different conditioning** — when the inputs differ across chosen and rejected sides, accepted posters repair the derivation (symmetric pairs, or one shared context so normalizers cancel) instead of bolting on a heuristic input-side term. Poster-only evidence. [ids: tIW29IpCwG, dtHEthIjmu, 7lpDn2MhM2]

## Failure modes (from Reject)
- **The contrastive term is a near-copy of an existing objective** — an input-side "prefer the intact input over a degraded copy" term, a symmetric objective, a curriculum with a refreshed reference, or adversarial hard negatives in a contrastive loss were each judged too close to prior methods. Without a new reason the pair is constructed this way, the work reads as integration. [ids: uZ5AmOJKqV, DztuuQCn13, VF5xa8k0G0, qqlWHIjJ24, oJoLQlnNiB, JhHEsrQKk5]
- **The corruption is not shown to mean what the paper claims** — a misleading-instruction flip rate that may measure instruction-following rather than uncertainty, capability-gap labels where both answers may be wrong, hand-synthesized wrong-knowledge trajectories never checked against real errors, a self-written consensus never shown to beat the teachers. When the negative's validity is untested, the whole training signal is suspect. [ids: 2OANNtX3T5, JhHEsrQKk5, qqlWHIjJ24, 03u504EDJp]
- **The induced failure is rare or artificial, and a trivial fix is not ruled out** — adversarial prefix dialogues judged rare in practice (a chat reset might suffice), template-based misleading prompts without a prompting baseline, synthetic distortion games without a prompting baseline, multiple-choice-only unsolvable variants with straightforward remedies. [ids: sw6Wpx2LGr, 2OANNtX3T5, AZghddZY0r, K4YMFdx2Z2]
- **The gain is confounded with the curated data or another stage** — strong numbers that may come from a 2M-triple curated set rather than the objective, ablations showing the supervised warm-start did most of the work, or a dependence on a separate candidate shortlist. Without a data-controlled comparison, the contrast is not credited. [ids: YVsiB41ifI, qqlWHIjJ24, oJoLQlnNiB]
- **Evaluation is too narrow to support a general claim** — one old backbone family, binary existence benchmarks only, multiple-choice only, or a single application domain, with marginal gains on modern models. [ids: uZ5AmOJKqV, VF5xa8k0G0, 49qqV4NTdy, K4YMFdx2Z2, 03u504EDJp, DztuuQCn13]
- **Benchmarking existing algorithms on the manufactured setting, with no method or takeaway** — a new contrast setting or a controlled ablation study is not enough if the findings are not stated clearly or no new mechanism comes with it. [ids: AZghddZY0r, 49qqV4NTdy, K4YMFdx2Z2]

## Oral vs Reject gap
Here the success side is just the two orals (no spotlights). Both use manufactured pairs as the *source of a capability*: a compact verifier trained on synthesized true/false pairs for a few atomic skills that transfer, or the model's own ability to compare near-duplicate inputs, used as a perceptual check that rewrites its rationales. Each also reports which pair types or skills work and which do not. The 13 rejects mostly use the same raw material (masked or degraded inputs, adversarial negatives, swapped knowledge, capability-gap pairs) only as one more term in an existing preference or contrastive loss. They seldom validate that the negative captures the intended failure, and they test on narrow, saturated or synthetic-only settings. So the observable gap is (a) a measured failure behind the corruption, (b) a contribution beyond the loss term (a transferable skill, a verifier, a new inference procedure), and (c) analysis of pair design. Posters show (a) and the minimal-difference localization often, and (b) less often. Because n_oral = 2 < 3, the oral-only version of this gap cannot be assessed. confidence: low

## Oral vs HC gap
Not assessed: HC labels were not assigned in this run (citation lookup incomplete); the reviewer-expectation panel carries the contrast.

## Reviewer expectations
- **Show that the manufactured negatives are valid: check quality and noise, ideally against human judgment** _(source: both)_ — orals were asked how noisy pairs or upstream noise affect results; rejects were faulted for unvalidated misleading prompts, noisy capability-gap labels and hand-built negatives. [ids: ZymCPON45y, DM0Y0oL33T, 2OANNtX3T5, JhHEsrQKk5, qqlWHIjJ24]
- **Separate the contribution from the closest prior contrastive or preference objective** _(source: both)_ [ids: ZymCPON45y, uZ5AmOJKqV, DztuuQCn13, VF5xa8k0G0]
- **Report the extra compute from generating pairs or running extra passes** _(source: both)_ [ids: DM0Y0oL33T, ZymCPON45y, VF5xa8k0G0, oJoLQlnNiB]
- **Evaluate beyond your own benchmark, one backbone family and binary error metrics** _(source: both)_ [ids: DM0Y0oL33T, uZ5AmOJKqV, VF5xa8k0G0, 49qqV4NTdy, K4YMFdx2Z2]
- **Give actionable insight on which skills or pair types transfer, and why** _(source: oral reviews)_ [ids: DM0Y0oL33T, ZymCPON45y]
- **Prefer a task-agnostic pipeline that scales without a reward model or human annotation** _(source: oral reviews)_ [ids: ZymCPON45y, DM0Y0oL33T]
- **Include the trivial baseline: prompting, a context reset, or supervised training on the same data** _(source: reject reviews)_ [ids: 2OANNtX3T5, AZghddZY0r, sw6Wpx2LGr, YVsiB41ifI]
- **Argue that the induced failure occurs in real use, not only under the synthetic corruption** _(source: reject reviews)_ [ids: sw6Wpx2LGr, AZghddZY0r, 2OANNtX3T5]

## Cognitive barriers
- The failure is blamed on model capacity or decoding when the cause is a training distribution with no negatives at all (only answerable, positive, single-input examples), so no one thinks to manufacture the missing side. [J44HfH4JCg, P9VdRQOyqu, uS4Wmg7PmE, K4YMFdx2Z2]
- It is assumed that preference labels must come from humans or a stronger judge. Seeing that the model's own induced failures, uncertainty, attention allocation or implicit reward can rank pairs means accepting that a weak model can label its own negatives. [49qqV4NTdy, FZW7Ctyjm3, lCTZsWotjL, zGTe0EwB8q, f7WBRSuf9l]
- Whole-sequence scoring hides that errors are local. Shared spans dilute the gradient, and one has to see that a controlled substitution isolates exactly the token or phrase to penalize. [yG1fW8igzP, fgfftCWPrn, iIbe6qDN0A, Zy2XgaGpDw]
- The input is treated as fixed context and all contrast is put on the output, so the policy's insensitivity to its evidence is invisible as a loss-level gap. [uZ5AmOJKqV, 7lpDn2MhM2, ekOwxTn65Y, izbBqTL8vb]
- The move often needs a role inversion that runs against the usual reading: a negative model trained to fail, teacher outputs used as negatives, a classical augmentation used as a preference generator, an adversarial objective recast as preference. A theoretical flaw (a partition-function mismatch) can also stay hidden because the loss still trains. [SF2GlFhVsS, 03u504EDJp, jrKj2G2bQ4, FEMv4lHJ2C, dtHEthIjmu, tIW29IpCwG]

## Representative examples
| forum_id | label | one-line what it does |
|---|---|---|
| DM0Y0oL33T | oral | Trains a small generative verifier on synthesized true/false pairs for a few atomic skills and turns its critiques into edits for sequential refinement |
| ZymCPON45y | oral | Pairs near-duplicate inputs with matched queries so the model contrasts them and rewrites its own hallucinated rationales, then fine-tunes on the refined rationales |
| yG1fW8igzP | poster | Swaps correct concepts for plausible absent ones and penalizes only the swapped phrases, with token-wise KL to a frozen reference |
| izbBqTL8vb | poster | Adds a policy-gradient term maximizing divergence between outputs on the original and a masked input, after an error audit shows perception errors dominate |
| Zy2XgaGpDw | poster | Perturbs known-correct text at templated error types to get free token-level labels for a per-token reward model |
| uZ5AmOJKqV | reject | Adds a "prefer intact over most-damaged masked input" preference term; judged close to an existing conditional objective, tested on old backbones |
| 2OANNtX3T5 | reject | Uses answer flips under misleading prompts as a cheap uncertainty probe; validity of the probe left unresolved |
| VF5xa8k0G0 | reject | Generates adversarial minimal-difference inputs as contrastive hard negatives; similar to prior work, binary benchmarks only, marginal gains |
