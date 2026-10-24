# Synthesize Supervision From a Privileged Exact Representation
_id: `privileged_proxy_synthesis`_

**Plain alias**. _Label from the answer key, train on the raw input_

**Definition**. Replace a hard-to-process input with an exact symbolic, textual or simulator-side stand-in available only at data-creation time, let a strong generator produce accurate supervision from it, and train the target model on the raw input.

**Operational signature**. obtain exact stand-in (render, simulate, annotate) → feed stand-in to strong generator → synthesize instructions/traces at scale → train target on raw input paired with them

**When to apply**. When ground truth is cheap to create programmatically but direct annotation of the raw input is expensive or unreliable.

**Evidence base**. primary-tagged n=86, oral 1, spotlight 3, poster 52, reject 30; supporting-only n=35. The success side (oral ∪ spotlight) has only 4 papers, and the corpus has 13 orals in total. Most success conditions below therefore lean on posters as accepted support, and each one says so.

## Success conditions (from Oral/Spotlight; Poster as support)
- **The quality of the stand-in-derived labels is measured against an independent reference, not taken on trust** — Accepted papers back their label quality with evidence that does not come from the generator itself. Holi-Spatial (spotlight) compares annotation quality with feed-forward and per-scene baselines. Voila-A (spotlight) adds a real-gaze test set next to its trace-derived proxy. Among posters, SCOPE reports 91–96% human acceptance of its evolved samples, the chest-X-ray scene graphs are scored against existing annotations, and AQuA adds human-agreement evidence in rebuttal. The stand-in is the whole basis for the labels, so showing that it is faithful is what turns "cheap labels" into "correct labels". [ids: UGAP2F6FfV, ZYrZ5V84ZI, bTRH1O5Bp2, LrmyW9JLYq, 7b1MpD6IF8]
- **Ground truth is exact by construction, and the paper says why** — Simulator metadata, rendering programs, state machines and page source give answers that are correct by definition rather than guessed by the generator. One spotlight (simulated scenes with exact answers) and many posters (rule-based diagram engine, FSM websites verified by graph reachability, page-structure parsing, training-only structured tables, game code mapped to checkable answers) use this lever. Exactness alone does not secure acceptance: several rejected papers had it too (see failure modes). [ids: SBYCu5uJJf, MnJzJ2gvuf, jBPFdqmOck, H8KE7sudq5, AuBSUgFVgq, e4FqU4SyHL]
- **Gains transfer beyond the generator's own distribution: other backbones, external benchmarks, real deployments** — The spotlights show broad gains across many models and benchmarks (SBYCu5uJJf) and downstream fine-tuning gains on a large released dataset (UGAP2F6FfV). Posters show zero-shot transfer from synthetic to real providers, robots or live sites, and gains on seven out-of-domain benchmarks. Transfer is the only evidence that the model learned from the raw input rather than from artifacts of the generator. [ids: SBYCu5uJJf, UGAP2F6FfV, HWTRHS26ZF, vSLzoUoJt6, HaIEP2PD4S, e4FqU4SyHL, PrM9JAYDF6]
- **The paper opens a capability or interaction format that had no training data, with a minimal model and a full release** — The single oral (LLaVA) was praised as pioneering. It used text-only authoring from captions and boxes, a simple connector, and released data, code and models. Voila-A (spotlight) was credited with a "novel problem" and new datasets. Posters on identity references and personalization follow the same template: a new format, simple training, and data as the contribution. [ids: w0H2xGHlkw, ZYrZ5V84ZI, N5YTixK4F1, sAxdIJ4l6z, c6XIVI3TiQ]
- **The privileged view is used only during training and is removed at inference, and the paper states this asymmetry** — Training-only structured tables, OCR scaffolding that the student no longer needs, scene graphs used only as a data prior, and zoom-isolated labels moved back onto the full image all make the training/test split of information the central claim. This set is all posters (the oral also trains on raw images paired with text-derived data), so treat it as accepted-tier support rather than a marker of top-tier acceptance. [ids: AuBSUgFVgq, Ygg3KJPeT6, hRSLIreHlx, bVXx0K0j4c, w0H2xGHlkw]
- **Matched controls isolate the stand-in's contribution from "more data" or "any extra signal"** — Several posters crossed the threshold only after rebuttal added matched ablations. Examples: motion-representation alternatives, a text-only control, and an ablation separating data from optimizer. This shows that reviewers accept the pattern once attribution is clean. [ids: Ax02eR2c3d, c6XIVI3TiQ, nHW64r5KFG, ISZPRsh5YV]

## Failure modes (from Reject)
- **Data from the stand-in presented as the whole contribution, assembled from known parts, with an expected result** — The most common reject reason. Examples: "more OCR data helps", relation dialogues on an unchanged architecture, tool-use data from an earlier pipeline, raw-data alignment already used in prior work, and an incremental extension of an existing instruction set. The pattern is easy to execute, so reviewers want an insight beyond the data recipe. [ids: tj4a1JY03u, EA8dTp96GY, IB1HqbA2Pn, 94LyPGDi0Y, kWGHZuW5yJ, R1NUjHPUQI, hqFNqVQEUd]
- **Circular evaluation: the same generator family writes the data, builds the test and acts as judge** — Rejected papers scored outputs with the model that authored the training data, or with a benchmark from the same pipeline. Reviewers read the gains as style imitation. [ids: DUkYDXqxKp, EA8dTp96GY, oq5EF8parZ, 68FgdHQvJH, u4PmZOmtko, 5YCdnxerit]
- **The stand-in is noisy but its error is never quantified or stress-tested** — Specialist-model pseudo-labels near 63% task accuracy, about 29% mean error in reconstructed motion, model judges that themselves score about 60%, unaudited rationales, and grafted labels with unanalyzed bias. When the "exact" representation is not exact, the premise of the pattern fails. [ids: nCZH28rKQ9, o04FuYtHiZ, jNmsuEE4Gf, XplQLo1I9K, NRY0QAvGNT, 063FuFYQQd]
- **Only synthetic or single-source evaluation, with no evidence of transfer to real inputs** — Purely rendered notation, synthetic-only distortions, a driving-dominated benchmark, a single simulator, and a single game. Reviewers suspect the model learned the renderer or simulator rather than the task. [ids: odEmejlOoJ, kWGHZuW5yJ, o04FuYtHiZ, G6DLQ40VVR, BXKgPhaOzr, qOLviGc6BG]
- **The paper never shows that the intermediate representation is necessary** — Reviewers asked why not feed the raw input to a strong model directly, compare against an equal-size generic dataset, or attach a simple tool front-end. Rejected papers had not run these comparisons. [ids: kGgHappUSR, cjlPAgNifc, DiRJUdmZoK, o04FuYtHiZ, EA8dTp96GY]
- **Template-generated tasks that test local recognition while the claims say "reasoning"** — Formulaic two-turn dialogues, local rule families described as "boundless reasoning", templated long-video questions, and qualitative-relation templates. Narrow generators invite overclaiming, and reviewers penalize the mismatch. [ids: oq5EF8parZ, 6EO8JzLyBS, RW7Z1W1Hux, 4Ha2srdhPN, DUkYDXqxKp]

## Oral vs Reject gap
On the success side (1 oral, 3 spotlights), method mechanics barely differ from the rejected papers. All four share the concerns reviewers raised against rejects: LLaVA's evaluation is weak and its data is unvalidated, Voila-A relies on GPT-based ranking, and Holi-Spatial has limited methodological novelty and possible leakage. The difference is in what surrounds the mechanics.

1. Success papers either open a capability that had no data at all (LLaVA, gaze conditioning) or pair exact-by-construction labels with independent quality evidence and broad cross-model, cross-benchmark transfer (Holi-Spatial, the simulation spotlight).
2. Rejected papers apply the same recipe to a known capability, judge with the generator family, leave stand-in noise unquantified, and evaluate on one synthetic or single-source domain.

The cleanest paired evidence is Voila-A. It was rejected at one venue (KrOmLMFYHi: "engineering-oriented novelty, unclear presentation, only 100 real gaze samples") and the same method became a spotlight at the next (ZYrZ5V84ZI: "novel problem, creative proxy, open synthetic and real-gaze datasets, extensive ablations"). So framing the problem as new and adding real-signal evidence moved the decision, while the mechanism stayed the same.

With only 1 oral (n_oral < 3), the oral-only version of this gap cannot be assessed.

## Oral vs HC gap
Not assessed: HC labels were not assigned in this run (citation lookup incomplete); the reviewer-expectation panel carries the contrast.

## Reviewer expectations
- **Validate the synthesized labels independently (human audit, expert check, or comparison with real annotations), not only through downstream accuracy** _(source: both)_ — It was raised against the oral ("data quality… unvalidated"), praised in a spotlight (annotation quality versus baselines), and cited in rejects. [ids: w0H2xGHlkw, UGAP2F6FfV, 063FuFYQQd, jNmsuEE4Gf, nCZH28rKQ9]
- **Keep the evaluator or judge independent of the model family that generated the training data** _(source: both)_ — The oral and a spotlight drew concern for language-model or GPT-based scoring, and rejects were sunk by full circularity. [ids: w0H2xGHlkw, ZYrZ5V84ZI, u4PmZOmtko, 68FgdHQvJH, EA8dTp96GY]
- **Compare against the obvious test-time pipeline that uses the stand-in or a tool directly** (for example, a description plus a text-only model, or a recognition front-end) _(source: both)_ [ids: w0H2xGHlkw, DiRJUdmZoK, o04FuYtHiZ, kGgHappUSR]
- **Check train/test overlap when one generator produces both splits** _(source: both)_ — Raised against a spotlight and in rejects. [ids: UGAP2F6FfV, oq5EF8parZ, cjlPAgNifc, RW7Z1W1Hux]
- **Reward scale, completeness and open release of the data engine** _(source: oral reviews)_ — This is the dominant praise on the success side. [ids: w0H2xGHlkw, UGAP2F6FfV, SBYCu5uJJf, ZYrZ5V84ZI]
- **Show a contribution beyond assembling existing components** _(source: both)_ — "Limited methodological novelty" appears even in spotlight reviews, but there it is outweighed by validated quality and transfer. In rejects it is decisive. [ids: UGAP2F6FfV, tj4a1JY03u, IB1HqbA2Pn, R1NUjHPUQI]
- **Address the gap between synthetic or simulated data and real inputs explicitly** _(source: both)_ [ids: SBYCu5uJJf, odEmejlOoJ, kWGHZuW5yJ, BXKgPhaOzr]

## Cognitive barriers
- A capability gap gets blamed on architecture or encoder capacity when the real cause is a missing supervision format (oq5EF8parZ, JKEIYQUSUc, SBYCu5uJJf, PrM9JAYDF6, EA8dTp96GY).
- Information that is unavailable at test time looks useless for training, so the training-only privileged view goes unused (AuBSUgFVgq, bVXx0K0j4c, Ygg3KJPeT6, U806q3iILo).
- The default is to ask a strong model to perceive the raw input and judge it directly. Conditioning that model on exact labels and letting it supply only the language is the less obvious choice (kWGHZuW5yJ, cjlPAgNifc, kGgHappUSR, dcLjhvVU47).
- Datasets are treated as fixed collected artifacts, and real data is treated as the necessary distribution, which hides the option of building a controllable environment whose ground-truth state is known (uimrBBfDCH, jBPFdqmOck, HaIEP2PD4S).
- Dense annotation often already exists in neighboring artifacts (captions and boxes, audio descriptions, paper captions, trace data) but is not recognized as a stand-in. A deliberately less precise but guaranteed-correct stand-in is likewise passed over (w0H2xGHlkw, RW7Z1W1Hux, DEOV74Idsg, ZYrZ5V84ZI, 4Ha2srdhPN).

## Representative examples
| forum_id | label | one-line what it does |
|---|---|---|
| w0H2xGHlkw | oral | Text-only generator writes instruction data from captions and boxes; a simple connector is trained on raw images |
| ZYrZ5V84ZI | spotlight | Mouse traces as a scalable gaze proxy for generated instruction data, plus a real-gaze test set |
| UGAP2F6FfV | spotlight | Chained foundation tools with confidence triage auto-annotate 3D scenes from raw video at 4M scale |
| SBYCu5uJJf | spotlight | Simulator metadata gives exact spatial answers; combined with structured scene-map prompting |
| AuBSUgFVgq | poster | Training-only structured tables let a reasoner write traces, which are transferred to an image-only model |
| KrOmLMFYHi | reject | Earlier submission of the gaze-proxy method; rejected for engineering novelty and small real evaluation |
| cjlPAgNifc | reject | Plotting code as a lossless bridge for chart QA synthesis; judged incremental and small-scale |
| u4PmZOmtko | reject | Educational videos turned into instruction data, with one LLM generating, benchmarking and judging |
