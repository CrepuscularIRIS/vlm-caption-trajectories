# Trajectory libraries A and B (public-safe export)

This folder publishes the two SFT trajectory libraries compared in [docs/22](../docs/archive/22_dual_library_eval_20261016.md): every assistant turn (short evidence caption, plan/NEXT text, crop action, final answer) and the tool receipts between them. Questions, options and images are **not** included. Field reference: [SCHEMA.md](SCHEMA.md). Numbers: [STATS.json](STATS.json).

| File | Records | Assistant turns | Notes |
|---|---:|---:|---|
| `library_A_v2_2.jsonl` | 4,040 | 7,047 (all supervised) | 7 generator families, mean 1.74 steps, 52.8% direct answers |
| `library_B_v4c.jsonl` | 3,983 | 14,826 (9,898 supervised + 4,928 forced-miss turns) | mean 3.72 steps, 17.0% direct answers, forced misses in 2,676 records |
| `system_prompts.json` | 2 | | Caption v5.1 system prompt (A) and caption v6 skills prompt (B) |
| `check_public_export.py` | | | Validator, see below |

## Library A: V2.2 KEEP (caption v5.1)

A is the frozen V2.2 release described in [docs/19](../docs/archive/19_v2_2_complete_design_and_catalog_20261009.md): one path per question, chosen from four main generators (Luna, Sol 5.6, GLM, Grok) plus targeted top-ups from Astra, Gemini and K3. Each generator answered with the "short evidence caption + real crop action" protocol at native resolution with a budget of 8 crops. Every selected path was reviewed view by view by a model reviewer and is KEEP. These are model reviews, not human gold labels. There are 4,040 tasks and 3,690 original-image groups. The paths contain 2,132 direct answers, 1,146 one-crop paths and 762 paths with two or more crops. The A SFT run in docs/22 did not use all 4,040 records. It used the 3,983 questions shared with B, giving 6,953 turn rows. `in_sft_aligned_3983` marks those records. The private aligned file was not available locally, so the flag is rebuilt from B's item list. The rebuilt subset has 6,953 assistant turns, which matches the training row count.

## Library B: v4c interpolated (caption v6)

B is the interpolated library described in docs/22 §2.3–2.4. It covers the same 3,983 questions as the aligned A subset. Generators (GPT-5.6 Sol 3,422, GPT-6.1 Sol 513, Astra 48) wrote trajectories under the caption v6 protocol. Each turn has SEE / KNOW / NEXT (or SO) lines and a skill word: locate, narrow, verify, shift, widen, scan, commit or abstain. Views were about 2 MP and the budget was 8 crops. On some turns where the generator itself asked for a crop, the constructor replaced the executed box with a forced miss: a shift, clip, tight crop or far crop. The generator then reacted to the view that came back. These forced-miss turns are kept in the history but carry no SFT loss. In this export they are marked with `induced_miss` on the turn and `supervised: false`, and the record-level `induced_steps`, `induced_kinds`, `reactions` and `assistant_supervision` fields show the construction. In total there are 4,928 forced misses across 2,676 records. B has had no semantic review (`semantic_review: NOT_DONE`).

## What was removed and why

The public-repo rule is no benchmark content beyond identifiers, and nothing from our machines.

- **Images**: no pixels, base64 or file paths. An image is identified only by `bench` + `item_id` and `image_sha256` of the original file. `<image>` placeholders in the text mark where views were inserted.
- **Question and options**: the first user turn begins with `Question: [question and options removed; ...]`. We keep `category`, `answer_kind`, `n_options` and `gold_letter`. Gold answers for `short`/`open`/`bbox` items are not published (A: 103 items, B: 90). The final `<answer>` the generator gave is kept.
- **Machine and provider details**: raw result paths, image paths, hostnames, endpoints and request logs are dropped. Only hashes of the private raw records remain.
- **Review prose**: A keeps `verdict`, `selected`, `phase` and `behaviors`. The Chinese one-line review summaries are left out because some of them quote option text. They are still in `web/data/catalog.json`.
- **System prompts**: each is stored once in `system_prompts.json` and is not repeated in every record.
- **Kept as model text**: assistant turns are verbatim. No assistant turn quotes a full question or question stem. Some turns quote one option text of 20 or more characters, usually the option being chosen or the "image does not feature..." option: 158 turns in A and 174 in B. Two more A turns quote an option line together with its label, such as "(D) ...". A few turns transcribe a URL or dotted number seen in the image (A: 3 URLs, 2 IPv4-like; B: 2, 4). They are reported, not removed.

## Re-attaching questions and images

Download the public benchmarks and map `item_id` as follows. Our loaders kept the original option order, so `gold_letter` refers to the benchmark's own options.

| `bench` / id pattern | Source | Mapping |
|---|---|---|
| `mme_realworld_NNNNN` | MME-RealWorld (`yifanzhang114/MME-RealWorld`, `MME_RealWorld.json`) | zero-based row index `NNNNN` in the JSON list; options = `Answer choices` |
| `mme_realworld_lite_N` | MME-RealWorld-Lite | the item whose `index` field is `N` |
| `mmmu_pro_std_<id>` | MMMU-Pro (`MMMU/MMMU_Pro`), config `standard (10 options)` | row with `id == <id>`; `<image 1>` in the question was rendered as "the image" |
| `worldbench_NNNN` | WorldBench (`worldbench-submission/WorldBench`, `train.parquet`) | zero-based row index `NNNN`; options A–D = `option_a`…`option_d` |
| `visreason_c<K>_<L>` | VisReason (`CASIA-IVA-Lab/VisReason`), `class_<K>.jsonl` | record with `label == <L>` (single-image items only) |
| `vs_NNN` | V* Bench (`craigwu/vstar_bench`, `test_questions.jsonl`) | `question_id == NNN` |

Check the image you download against `image_sha256`. The first user turn records the original pixel size and the scale it was shown at. Each tool receipt gives the crop's region in original-image normalised coordinates, so every observation can be rebuilt from the original image.

## Validation

```bash
python trajectories/check_public_export.py            # structural, string and count checks (public)
python trajectories/check_public_export.py --pool <pool.jsonl> --source-a <caption_sft_v2_2_keep.json> --source-b <DATABASE.jsonl>
```

The first command runs without private data. It checks record and turn counts, step/crop/supervision consistency, forbidden keys at any depth, path/URL/base64/key/endpoint patterns, the 50 MB file limit and agreement with STATS.json. The second form also scans every source question, stem, displayed prompt line and option text of 20 or more characters as an exact substring against all exported strings. It fails on any hit outside assistant text, and checks that assistant-text quote counts equal `STATS.json → question_quotes`.
