# Align Internal Representations to a Teacher
_id: `distill_internal_property`_

**Plain alias**. _Copy what the teacher has inside, not just what it says_
**Definition**. Identify a property missing from a model's intermediate representations or attention and add an auxiliary objective (or small adapter) that distills it from a specialist or context-conditioned teacher alongside the main objective.
**Operational signature**. probe internals for the missing property (with a proxy that tracks downstream performance) → choose a teacher that has it (a specialist, or the model itself under privileged context) → add an auxiliary alignment loss or removable adapter → train jointly while limiting drift → drop the teacher at inference
**When to apply**. When output-level supervision is insufficient and the deficiency is visible in internal states that a teacher already gets right, especially when feeding the teacher at inference is too costly.
**Evidence base**. primary-tagged n=31, oral 1, spotlight 1, poster 20, reject 9; supporting-only n=15

Note: the success side has only 2 papers (1 oral, 1 spotlight; orals are 13 in the whole corpus). Posters (n=20) are used below as supporting accepted evidence and are marked as such.

## Success conditions (from Oral/Spotlight; Poster as support)
confidence: low (only 2 oral/spotlight papers; each condition rests mainly on poster support)
- **The paper reframes what the transferable property is, showing it is already latent or compressible** — The two success papers do not just add a distillation loss; they first argue a new account of where the capability lives: reasoning text as a compact, domain-aligned representation that a small contrastive model can be aligned to, and geometry already present in a frozen backbone's tokens that only needs eliciting. Posters show the same move when the teacher is the model itself under privileged conditioning (its own text-input policy, its own grounded rollout, its own demonstration-prompted pass, its own milder-compression pass). [ids: DkHt2K1g2Y, CHavqrN1X9; poster support: MrrLfshRVj, UzusuBuNfT, QhRemVrZbG, gZjPllL9jM]
- **The teacher or extra module is removable, so inference cost falls or stays flat** — Success papers sell a deployment asymmetry: about 4% of parameters and runtime with no language model at inference, or a zero-initialized side module that can be attached and removed. Posters get credit for the same asymmetry (expert features distilled into hidden layers instead of extra encoders, token budgets cut by ~95%, demonstrations removed at inference). [ids: DkHt2K1g2Y, CHavqrN1X9; poster support: AZ1iyo58F8, KQuVaVUPL6, 5gLgjn3MVT, rDKFflrjZK]
- **A measurable internal proxy is linked to downstream performance before the loss is added** — Accepted papers motivate the auxiliary objective with an internal measurement that tracks the target capability (attention and embedding-space analysis; multi-view correspondence quality; probe-measured representation quality; teacher-student attention similarity; layer-wise hidden-state shift at the layers that trigger a behavior; the teacher's sharper, lower-entropy attention). [ids: DkHt2K1g2Y; poster support: KQuVaVUPL6, AZ1iyo58F8, Wa9Bg9b50B, 45rvZkJbuX, DVW16DW1Cn]
- **Drift on the original capability is limited by construction** — The spotlight freezes the backbone and zero-initializes a side path; posters add a small aligner to avoid forgetting, align only at specific layers, or update only the unprivileged path, and report preserved general ability. [ids: CHavqrN1X9; poster support: DVW16DW1Cn, 45rvZkJbuX, fSE0rUngCX, 19Wap8LePy]
- **Gains hold across several backbones and benchmark families** — Both success papers are praised for breadth (diverse datasets; perception, understanding and embodied transfer), and the strongest posters show consistent gains across encoders, language models and taxonomies. [ids: DkHt2K1g2Y, CHavqrN1X9; poster support: KQuVaVUPL6, 8q9NOMzRDg, 19Wap8LePy]

## Failure modes (from Reject)
- **A standard distillation target is adapted with no new account of what should be transferred** — Rejected papers distill hidden states, logits, attention maps, inter-token relations or expert features using known objectives, and reviewers read them as "known techniques plus an extra stage" whose findings confirm expected behavior. [ids: 774F8gF0UO, MFySy0DWAH, Qvx7G5Fsy0, NKa11pYuWp, ypptmENxJn, IfyaPN8rwX]
- **A close prior method already does nearly the same alignment** — Several rejects are judged a simplified or incremental variant of a specific published system, or their core correlation is already established, which reviewers treat as decisive. [ids: IfyaPN8rwX, ypptmENxJn, Qvx7G5Fsy0, J2WeAMiSZh]
- **Comparisons are unfair or omit the distillation literature** — Methods use inputs baselines lack, compare against differently trained models, skip equal-compute vanilla distillation, or omit a competitor's key component, so the gain cannot be attributed to the transferred property. [ids: 7wX5uL459y, IfyaPN8rwX, ypptmENxJn, MFySy0DWAH, lja4JMesmC]
- **Narrow scope with a general framing** — Single benchmark, single domain, one task type or two older models, while the title or claims speak about general adaptation or compression. [ids: lja4JMesmC, 7wX5uL459y, 774F8gF0UO, MFySy0DWAH, ypptmENxJn]
- **Gains are marginal relative to the added cost or complexity** — Extra modules, stages or teacher passes buy fractions of a point (e.g. one component adding ~0.3), or barely beat the small specialist they distill from, without cost accounting. [ids: lja4JMesmC, IfyaPN8rwX, NKa11pYuWp, ypptmENxJn]
- **The alignment is placed where practitioners cannot use it** — The fix requires repeating pretraining and fails on already fine-tuned models, or depends on a heavy pseudo-label pipeline built from proprietary models with no label-quality analysis. [ids: J2WeAMiSZh, NKa11pYuWp]

## Oral vs Reject gap
confidence: low (success side n=2)
The two success papers and the rejects use similar machinery (an auxiliary alignment loss or side module trained against a teacher), so the gap is not technical. The success papers lead with a new claim about where the capability lives (reasoning output as a compact representation; geometry latent in frozen tokens) and turn it into a clear deployment asymmetry (the teacher or language model disappears at inference, or a removable side path leaves the backbone untouched). Rejects instead pick a familiar target (attention, hidden states, relations, expert features), refine how it is matched, and are then judged against a close prior method, on unfair or narrow comparisons, and on marginal gains relative to cost. The poster tier sits in between: posters typically add the internal-proxy-to-performance link or a self-conditioned teacher, but without a reframing claim, and one reject reappears later as a poster with the same method (IfyaPN8rwX→gJy2pIp7gt), so the poster/reject boundary is noisy. With only 1 true oral (DkHt2K1g2Y), the oral-only version of this gap cannot be assessed.

## Oral vs HC gap
Not assessed: HC labels were not assigned in this run (citation lookup incomplete); the reviewer-expectation panel carries the contrast.

## Reviewer expectations
- **Large inference-time savings at matched quality, from a design that is easy to attach or remove, are a primary reason to accept** _(source: oral reviews)_ [ids: DkHt2K1g2Y, CHavqrN1X9]
- **Show causally, not just by correlation, that the distilled property is what drives the gain (elicited rather than newly learned; ablations rather than single-example attention)** _(source: oral reviews)_ [ids: DkHt2K1g2Y, CHavqrN1X9]
- **Do not overstate "no extra prior" or "retained general capability"; report utility and capability retention fully** _(source: both)_ [ids: CHavqrN1X9, J2WeAMiSZh]
- **Address teacher dependence: sensitivity to teacher quality, whether larger teachers help, and the quality of teacher-generated or pseudo labels** _(source: both)_ [ids: DkHt2K1g2Y, MFySy0DWAH, NKa11pYuWp]
- **Test across several backbones, benchmarks and domains** _(source: both)_ [ids: CHavqrN1X9, DkHt2K1g2Y, lja4JMesmC, 7wX5uL459y, 774F8gF0UO]
- **Position sharply against the closest prior alignment or distillation method and compare to it under its full configuration** _(source: reject reviews)_ [ids: IfyaPN8rwX, ypptmENxJn, Qvx7G5Fsy0, J2WeAMiSZh]
- **Compare against established distillation baselines under identical inputs and equal compute** _(source: reject reviews)_ [ids: MFySy0DWAH, 7wX5uL459y, IfyaPN8rwX]
- **Report training cost and overhead of the auxiliary branch or teacher passes** _(source: reject reviews)_ [ids: lja4JMesmC, IfyaPN8rwX, Qvx7G5Fsy0]

## Cognitive barriers
- An output is usually seen only as an output: reasoning text as an explanation whose length cost is unavoidable, attention as an interpretability by-product, coordinates as the only grounding target. Seeing these as representations that can serve as distillation targets or supervisable signals is the step most papers credit as non-obvious (DkHt2K1g2Y, NKa11pYuWp, ypptmENxJn).
- When a capability is weak, the default assumption is that it is missing, so the remedy is to inject inputs or fuse extra encoders; the alternative, that it is latent and under-elicited, opens the cheaper plug-in or training-only route (CHavqrN1X9, AZ1iyo58F8, Qvx7G5Fsy0, KQuVaVUPL6).
- Earlier reports that intermediate-feature, attention or pixel-level alignment fails discourage trying structured internal targets at all; the barrier is to locate the right target (latent tokens, instruction-salient attention, transitions) rather than abandon internal alignment (gJy2pIp7gt, IfyaPN8rwX, 8q9NOMzRDg).
- The teacher need not be a different model: the same model under privileged conditioning (text input, demonstrations, grounded outputs, milder compression) already has the property, which is easy to miss when distillation is framed as big-to-small (MrrLfshRVj, UzusuBuNfT, QhRemVrZbG, rDKFflrjZK, gZjPllL9jM).
- Which internal level carries the transferable property is not obvious: shallow cross-input attention, a generative model's word-to-region attention, or the specific layers that trigger a behavior can be better carriers than logits or deep features (7wX5uL459y, DVW16DW1Cn, 45rvZkJbuX).

## Representative examples
| forum_id | label | one-line what it does |
|---|---|---|
| DkHt2K1g2Y | oral | Explains a reasoning model's generalization as compression into a short text representation and distills that text space into a small contrastive model with no language model at inference. |
| CHavqrN1X9 | spotlight | Attaches a removable zero-initialized side module to a frozen backbone and trains it with dense teacher-derived geometry targets to elicit latent spatial knowledge. |
| KQuVaVUPL6 | poster | Links multi-view correspondence quality to downstream 3D performance, then aligns visual features to a frozen geometry teacher alongside the task loss. |
| MrrLfshRVj | poster | Uses the model's own text-input policy as a teacher and matches per-step distributions so reasoning transfers to an equivalent rendered input. |
| DVW16DW1Cn | poster | Distills a generative model's sharper word-to-region attention into an understanding model's attention via an auxiliary loss and small aligner. |
| IfyaPN8rwX | reject | Adds instruction-to-input attention alignment and token-transition alignment to distillation; judged incremental versus close prior work (later accepted as a poster). |
| J2WeAMiSZh | reject | Pulls new-input embeddings toward native-text embeddings during pretraining to transfer robustness; core link judged already known and the fix impractical. |
| Qvx7G5Fsy0 | reject | Trains query tokens to match multiple frozen expert features with masked attention; judged too similar to a published system on outdated architectures. |
