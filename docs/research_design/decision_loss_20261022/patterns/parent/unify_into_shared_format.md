# Recast Everything Into One Shared Format
_id: `unify_into_shared_format`_

**Plain alias**. _Make all inputs and actions speak one language_
**Definition**. Convert heterogeneous signals, tasks, environments or action spaces into a single representation the base system already consumes or can emit, then train one model on the unified data.
**Operational signature**. inventory heterogeneous sources → define one canonical input/action format → convert or render all data into it → train a single model (often grounding first, then planning)
**When to apply**. When many related tasks or interfaces fragment the data and each is too small to train a specialized model.
**Evidence base**. primary-tagged n=39, oral 0, spotlight 5, poster 25, reject 9; supporting-only n=14

_Note on the evidence: there are no orals for this parent (the whole corpus has only 13); the success side is 5 spotlights. Poster papers are used as supporting accepted evidence where marked. The reject side is small (9 papers). HC labels were not assigned in this run._

## Success conditions (from Oral/Spotlight; Poster as support)
- **The unified format needs no architectural change and is shown to preserve the base system's existing abilities while adding cross-setting transfer** — success-side papers treat several input settings as special cases of one sequence template, or fold specialized tricks into text-formatted auxiliary tasks, and report retained original performance plus emergent transfer between settings. Posters that rewrite actions or masks in the model's native vocabulary earn the same praise when they show no forgetting. [ids: oSQiao9GqB, oHbVboLXz6, 8VVbElV06v; poster support: sFO9d6XSlf, vkakKdznFS, yaQbTAD2JJ]
- **The unification removes concrete inconsistencies across sources, such as conflicting labels for equivalent actions or incompatible task taxonomies** — the strongest papers name the specific conflict (platform-specific action names, siloed sub-domains with different task formats) and show that one canonical vocabulary or hierarchical definition system resolves it, so pooling data adds signal rather than noise. [ids: n9PDaFNi8t, 8VVbElV06v; poster support: PlihOwfx4r, ZKE23BBvlQ, L6bEitSMeu]
- **A large open resource (toolkit, converted corpus, benchmark, models) anchors the contribution** — every spotlight is accepted largely on releasing the infrastructure that makes the unified format usable by others; reviewers rate the resource as the main contribution even while calling the method light. [ids: n9PDaFNi8t, 6iRZvJiC9Q, oSQiao9GqB, 8VVbElV06v; poster support: ZKE23BBvlQ, L6bEitSMeu]
- **Ablations tie each unified component or auxiliary task to a specific gain** — accepted papers show which converted task, reasoning augmentation or training stage drives which improvement; this is what lets "we pooled data into one format" read as a finding rather than consolidation. [ids: oHbVboLXz6, 6iRZvJiC9Q, oSQiao9GqB; poster support: PlihOwfx4r, L6bEitSMeu, yaQbTAD2JJ]
- **One unified model matches or beats specialists and larger general models, and keeps improving as unified data grows** — success-side results are framed against both specialized and larger models, and data-scaling curves are offered as evidence that the shared format is the lever. [ids: oHbVboLXz6, n9PDaFNi8t, 6iRZvJiC9Q; poster support: iVxxgZlXh6, efFmBWioSc]

## Failure modes (from Reject)
- **Consolidating existing datasets into one format (often with a stronger backbone) without ablating the unification choices themselves** — reviewers ask for unified versus per-source formats, joint versus separate training and staged versus single-stage training; absent these, the work reads as system integration of known ingredients and gains are attributed to data volume or backbone. [ids: FHtHH4ulEQ, pZQvv5C7WL, hVwS9KkY6V, MwR1Uihush, SNGANmQPLv, 58681hX7oH]
- **Converted or synthesized training data whose quality is not measured** — model-generated instructions, plans or reasoning traces with visible errors, or a dataset produced mostly by a single model with no human check, undermine the whole unified corpus. [ids: 4QaKdsh15T, 9syH2SseOy, 58681hX7oH, FHtHH4ulEQ]
- **Weak or under-baselined results against recent or specialist systems** — a unified model that trails recent baselines, or is compared only with outdated or weaker-backbone systems, cannot justify the generalist framing. [ids: hVwS9KkY6V, 4QaKdsh15T, CcQFKeUK5K, pZQvv5C7WL]
- **Overstated claims of being first, generalist, on-device or inherently capable** — claims that the shared interface reveals inherent capacity, or that the model is the first or a true generalist, are checked against prior work or missing measurements and count heavily against the paper. [ids: 4QaKdsh15T, hVwS9KkY6V, pZQvv5C7WL, 58681hX7oH, CcQFKeUK5K]
- **The stated motivation for the shared interface is never directly tested** — e.g. the claimed harm of special tokens, the benefit of a task-definition prefix on its own, or the need for a reward stage over a supervised loss in the native format, is asserted but not ablated. [ids: 9syH2SseOy, SNGANmQPLv, CcQFKeUK5K]
- **Artifacts not released or incomplete, or no rebuttal** — for a contribution whose value is the converted resource, missing data, checkpoints or code (or silence in rebuttal) removes the main reason to accept. [ids: 58681hX7oH, hVwS9KkY6V, CcQFKeUK5K, pZQvv5C7WL]

## Oral vs Reject gap
The 5 spotlights and the 9 rejects use the same move and receive the same "limited methodological novelty" criticism; what separates them is what surrounds the shared format. Success-side papers release the conversion infrastructure and data at scale, name the concrete cross-source inconsistency the format removes, ablate each converted component, and compare against both specialists and larger models. Rejects consolidate data into one format without ablating that choice, rely on unmeasured model-generated data, compare against weak or outdated baselines, overclaim, or withhold artifacts. Two resubmission pairs in the dossier make the contrast concrete: the same unified interface-agent recipe was rejected for "missing ablations for the claimed designs" (FHtHH4ulEQ) and later accepted with "thorough ablations on training stages, inner monologue and cross-platform benefits" (PlihOwfx4r), and the same text-to-point grounding idea moved from reject (9syH2SseOy) to poster (aOxTMMLYuP) once the dataset pipeline was transparent and human-verified. With no orals for this parent (n_oral = 0), the oral-only version of this gap cannot be assessed.

## Oral vs HC gap
Not assessed: HC labels were not assigned in this run (citation lookup incomplete); the reviewer-expectation panel carries the contrast.

## Reviewer expectations
_"oral reviews" here means success-side reviews; for this parent that is spotlight reviews only._
- **Ablate the unification itself: unified versus per-source format, joint versus separate training, and the order of stages** _(source: both)_ [ids: 8VVbElV06v, oSQiao9GqB, FHtHH4ulEQ, pZQvv5C7WL, hVwS9KkY6V, SNGANmQPLv]
- **Verify the quality of converted and synthesized data at scale (error rates, human verification, judge reliability)** _(source: both)_ [ids: n9PDaFNi8t, 6iRZvJiC9Q, 8VVbElV06v, 4QaKdsh15T, 9syH2SseOy, 58681hX7oH]
- **Release the toolkit, data and models; open resources are the most praised element, and missing release is penalized** _(source: both)_ [ids: n9PDaFNi8t, 6iRZvJiC9Q, 8VVbElV06v, oHbVboLXz6, 58681hX7oH, hVwS9KkY6V]
- **Compare fairly against specialist models and the strongest recent baselines, under matched training conditions** _(source: both)_ [ids: n9PDaFNi8t, oHbVboLXz6, hVwS9KkY6V, 4QaKdsh15T, CcQFKeUK5K, pZQvv5C7WL]
- **Show generalization beyond the in-domain training mix and beyond one base model (e.g. leave-one-dataset-out)** _(source: both)_ [ids: oHbVboLXz6, oSQiao9GqB, CcQFKeUK5K, 4QaKdsh15T]
- **Limited method novelty is tolerated when the unified resource and results are strong** _(source: oral reviews)_ [ids: n9PDaFNi8t, oSQiao9GqB, 6iRZvJiC9Q, oHbVboLXz6, 8VVbElV06v]
- **Do not overclaim "first", "generalist" or deployment readiness without the corresponding evidence** _(source: reject reviews)_ [ids: 4QaKdsh15T, hVwS9KkY6V, pZQvv5C7WL, 58681hX7oH]
- **Report the inference and training cost the shared format adds (longer outputs, extra reasoning, multi-pass use)** _(source: both)_ [ids: oHbVboLXz6, 58681hX7oH, 9syH2SseOy, CcQFKeUK5K]

## Cognitive barriers
- Each setting, platform or sub-domain is handled by its own specialized model, pipeline and evaluation convention, which hides that they share one structure that a single format already covers. [ids: oSQiao9GqB, khAE1sTMdX, ZKE23BBvlQ, 8VVbElV06v]
- The performance gap is attributed to model capacity, scale or architecture rather than to inconsistent labels, data formats or a mismatch between the target representation and what the pretrained model already predicts. [ids: n9PDaFNi8t, 5KojubHBr8, sFO9d6XSlf, ZKE23BBvlQ]
- Some outputs (dense masks, continuous actions, 3D targets) look incompatible with plain token generation, so decoders, special tokens or new objectives are assumed necessary and the native-vocabulary encoding is not tried. [ids: vkakKdznFS, CcQFKeUK5K, yaQbTAD2JJ, aOxTMMLYuP]
- Improvements are habitually built as isolated architectural modules or new tokenizers, so recasting them as data-level auxiliary tasks in one training mix runs against field practice. [ids: oHbVboLXz6, iVxxgZlXh6, sFO9d6XSlf]
- Pooling across sources demands unglamorous engineering that did not exist — a consistent action vocabulary, synthesized step-level reasoning, a capture tool — so the payoff of unification is invisible until that infrastructure is built. [ids: FHtHH4ulEQ, PlihOwfx4r, 6iRZvJiC9Q]

## Representative examples
| forum_id | label | one-line what it does |
|---|---|---|
| n9PDaFNi8t | spotlight | Synthesizes cross-platform grounding data, pre-trains on it, then tunes on actions in a unified action space that resolves name conflicts. |
| oSQiao9GqB | spotlight | Treats multi-image, video, 3D views and patches as one interleaved template and jointly fine-tunes with no architecture change. |
| oHbVboLXz6 | spotlight | Folds separate specialized fixes into text-formatted auxiliary tasks in one multi-task fine-tuning mix for a generalist model. |
| 6iRZvJiC9Q | spotlight | Open tool and corpus of human demonstrations converted into state-action pairs with reflective reasoning; performance scales with data. |
| sFO9d6XSlf | poster | Rewrites low-level actions as hierarchical natural language the backbone already predicts, so a low-rank update learns control without forgetting. |
| PlihOwfx4r | poster | Unified pluggable action vocabulary plus synthesized inner monologue, grounding then planning; accepted resubmission with full ablations. |
| FHtHH4ulEQ | reject | Earlier version of the same unified agent recipe; rejected for missing ablations of the unified action space, multi-platform data and two-stage training. |
| hVwS9KkY6V | reject | Composes pretrained components and trains one stage on a mixed instruction set with task tokens; judged a swapped-component prior design with weak results. |
