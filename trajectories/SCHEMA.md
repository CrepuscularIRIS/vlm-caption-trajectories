# Schema / 字段说明

One JSON object per line, one line per question (task). 每行一个 JSON 对象，对应一道题。

- `library_A_v2_2.jsonl`: 4,040 records / 条
- `library_B_v4c.jsonl`: 3,983 records / 条
- `system_prompts.json`: the system prompt of each library, stored once and referenced by `system_prompt_id`. 每个库的系统提示只存一次，记录里用 `system_prompt_id` 引用。

## Item fields (both libraries) / 题目字段（两库共有）

| Field | Type | English | 中文 |
|---|---|---|---|
| `library` | str | `A_v2_2_keep` or `B_v4c_interp` | 所属库 |
| `audit_id` | str | Internal path id (A: `F#####`, B: `S####`); matches `web/data/catalog.json` for A | 内部路径编号；A 与网页目录一致 |
| `bench` | str | `mme_realworld`, `mmmu_pro`, `worldbench`, `visreason`, `vstar` | 题源 benchmark |
| `item_id` | str | Benchmark item id; see README for how to map it back | 题号，映射方法见 README |
| `category` | str/null | Benchmark category or subtask as recorded at generation time | 题目类别/子任务 |
| `source_dataset` | str/null | MME-RealWorld sub-dataset when present | MME-RealWorld 子数据集 |
| `source_version` | str/null | `full` for MME-RealWorld full | MME-RealWorld 版本 |
| `answer_kind` | str | `letter`, `short`, `open`, `bbox` | 答案类型 |
| `n_options` | int/null | Number of options (multiple choice) | 选项个数 |
| `gold_letter` | str/null | Gold option label when `answer_kind = letter` (usually A–J; a few VisReason items use digit labels); null otherwise | 标准答案选项标签；非选择题为 null |
| `image_sha256` | str | SHA-256 of the original benchmark image file | 原图文件 SHA-256 |
| `pool_split` | str | Our image-group split of the item pool (`train`/`dev`/`eval`) | 题池按原图组划分 |
| `material` | str | Coarse material class (`text`, `chart_table`, `natural`, `count`, `academic`, `visual_puzzle`, `map`) | 粗粒度素材类别 |
| `canonical_group_ids` | list[str] | Original-image group ids (same/near-duplicate images share a group) | 原图组编号 |
| `generator_model` | str | Model that generated the trajectory | 生成模型 |
| `protocol` | str | Caption protocol version | Caption 协议版本 |
| `steps` | int | Number of assistant turns | 助手轮数（步数） |
| `crops` | int | Number of crop actions | 裁图次数 |
| `direct_answer` | bool | `steps == 1` (answered without cropping) | 是否直答 |
| `final_answer` | str | Content of the final `<answer>` | 最终答案 |
| `system_prompt_id` | str | Key into `system_prompts.json` | 系统提示编号 |
| `turns` | list | Conversation without system prompt; see below | 对话（不含系统提示），见下 |

## Library A only / 仅 A 库

| Field | English | 中文 |
|---|---|---|
| `in_sft_aligned_3983` | True if the item is one of the 3,983 questions shared with B; the A SFT run used only these (6,953 assistant turns) | 是否属于与 B 相同的 3,983 题；A 组 SFT 实际只用这部分（6,953 个助手轮） |
| `generator_family` | Short family name used in the reports (Luna, Sol 5.6, GLM, Grok, Astra, Gemini, K3) | 报告里的模型简称 |
| `generator_effort` | Reasoning-effort setting recorded in the result | 推理强度设置 |
| `generator_demo_kind` | Kind of in-context demonstration shown to the generator (`direct`/`binding`/null); demonstrations are not in the export | 生成时给的示范类型；示范本身不在导出中 |
| `source_version_tag` | Release that admitted the path (`V1`, `V2_increment`, `V2.1_topup`, `V2.2_topup`) | 入选批次 |
| `result_sha256`, `review_record_sha256` | Hashes of the private raw result and review record | 私有原始结果与审核记录的哈希 |
| `review` | `verdict` (all KEEP), `selected`, `phase`, `behaviors` from the public catalog; the Chinese one-line summary stays in `web/data/catalog.json` | 审核结论与行为标签；中文一句话摘要见网页目录 |

## Library B only / 仅 B 库

| Field | English | 中文 |
|---|---|---|
| `split` | Split label written by the v4c constructor | 构造器记录的划分 |
| `campaign` | Generation campaign name | 生成批次名 |
| `accepted_by` | Rule that accepted the final answer (`frozen_scorer`, `open_normalized_equal`, `bbox_iou`, ...) | 答案判对规则 |
| `budget`, `arm`, `variant` | Crop budget (8); prompt arm (`verify_each_fact_v6` or `plain_v6`); view-size variant (`ctrl_2mp_b8`) | 裁图预算、提示分组、视图尺寸 |
| `source_result_sha256` | Hash of the private raw result | 私有原始结果哈希 |
| `skills` | Skill word of each assistant turn | 每轮技能名 |
| `see_verdicts` | `hit`/`partial`/`miss` from the SEE line (null at step 1) | 每轮 SEE 判词 |
| `induced_misses`, `induced_kinds`, `induced_steps` | Number, kinds (`shift`, `clip`, `tight`, `far`) and steps of forced misses | 强制失误的次数、类型、所在步 |
| `reactions` | Per forced miss: `step`, `kind`, intended skill it should elicit (`elicits`), target `coverage` of the returned view, and the generator's actual `next_verdict` / `next_skill` | 每次强制失误的设计反应与实际反应 |
| `assistant_supervision` | Per assistant turn, whether it carries SFT loss (false = forced-miss turn) | 每轮是否计 loss |
| `objective_candidate`, `semantic_review` | Constructor flags; `semantic_review` is `NOT_DONE` for all | 构造器标记；均未做语义审核 |

## Turns / 轮次

| Role | Fields | English | 中文 |
|---|---|---|---|
| `user` | `step`, `text` | First turn only. The question and options are replaced by a placeholder; the original-image size line, `<image>` placeholder and step header are kept | 仅首轮。问题与选项替换为占位符，保留原图尺寸、`<image>` 占位和步数提示 |
| `assistant` | `step`, `text`, `skill`, `action`, `bbox`, `source`, `answer`, `supervised` | `text` is the verbatim model turn: `<think>Step N \| SKILL ...</think>` then `<grounding>{"bbox_2d":[x1,y1,x2,y2],"source":VIEW}</grounding>` or `<answer>...</answer>`. `bbox` is normalised [0,1] in the `source` view. `action` is `crop` or `answer` | `text` 为模型原文；`bbox` 为相对 `source` 视图的归一化坐标 |
| `assistant` (B) | `induced_miss` | null, or `{kind, elicits, coverage, next_verdict, next_skill}` when this turn's crop was replaced by a forced miss (the turn is then `supervised: false`) | 该轮裁图被换成强制偏框时给出失误信息，此轮不计 loss |
| `tool` | `after_step`, `text` | Environment receipt after a crop: new view id, its region in original-image normalised coordinates, source/display size, `<image>` placeholder, next step header, remaining crops. No pixels | 裁图回执：新视图、在原图中的归一化区域、尺寸、`<image>` 占位、剩余次数；不含像素 |
