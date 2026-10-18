# 轨迹构造、审核与筛选代码（库 A / 库 B）

本目录是训练之前的一半：从公开基准题池出发，用多个教师模型在**真实裁图环境**里生成 caption + 裁图轨迹，再审核、筛选，装配成库 A（V2.2，4,040 条）和库 B（v4c 插值，3,983 条）。之后的 SFT 导出、GRPO 和评测见 [`../README.md`](../README.md)。

- 代码按实际运行的版本收录，只做了路径替换，见 [SCRUB_NOTES.md](SCRUB_NOTES.md)。`${...}` 是占位符，运行前要换成本机路径。
- `src/` 保持原来的平铺结构，脚本之间按文件名互相 import。收录范围是下表各入口脚本的完整本地依赖闭包，共 69 个文件。
- 不含生成结果、模型原始回复、图像与题干。轨迹本身见 [`../../trajectories/`](../../trajectories/README.md)。
- 文件名带日期或 `v4c`、`v3r2` 的是当时冻结的版本。后一版通常 import 前一版、只改一处，所以保留了整条链。

## 流程

| 阶段 | 做什么 | 入口（`src/` 下，除非注明） |
|---|---|---|
| 0 题池 | 合并公开基准训练部分，保留已封存划分；校验并重新抽取 MME-RealWorld；确定性评分（无模型裁判） | `funnel_pool.py`、`reconcile_corpus_splits.py`、`mme_rw_reextract.py`、`corpus_scoring.py`、`rescore_open.py`、`scoring.py` |
| 1 协议 | 系统提示、动作格式、裁图回执。库 A 用 v5.1（证据更新契约），库 B 用 v6（技能卡 + 三行 caption）；更早的 Mini-o3 冷启动复现与行为提示也在此 | `caption_v51.py`、`caption_v3.py`、`caption_v6.py`、`behavior_prompts.py`、`minio3_prompts.py`、`run_minio3_replica.py`、`monitor_step.py` |
| 2 生成（库 A） | 每个教师模型对题池逐题跑真实裁图 rollout，保存每一步的请求、回复和裁出的图；多模型（GLM、Luna、Grok、Sol 5.6、Gemini 等）分别成 run | `funnel_gen.py`、`funnel_gen_extra.py`、`run_caption_v51.py`、`models_v51.py`、`trajgen_models.py`、`tiered_transport.py`、`probe_api.py`、`probe_replica_api.py`、`glm_direct.py`、`cursor_anthropic.py`、`local_vlm.py`、`caption_generation_v2.py`、`caption_tiered.py`、`funnel_supplement_queue.py` |
| 3 审核 | 盲读事实（读图回答关键事实，不给答案）+ 过程审核（结构、对象绑定、每次裁图是否带来新证据）+ 多模型一致性；评分器自查 | `funnel_audit.py`、`api_reader.py`、`agy_reader.py`、`cc_proxy_reviewer.py`、`audit_v51.py`、`audit_tiered.py`、`consensus_pipeline_step.py`、`scorer_audit.py`、`export_protocol_check.py` |
| 4 筛选（库 A） | 逐轨迹特征与每题共识；每题只留一条完整轨迹；4,000 题选择与多步优先替换方案 | `funnel_select.py`、`build_selection_4k.py`、`export_selection_4k.py`、`build_selection_multistep_first.py`、`select_tiered.py`、`funnel_stats_report.py` |
| 5 装配库 A | V2：冻结去重队列 → 合并独立审核与历史 → 只导出严格 KEEP → CPU 校验与 LLaMA-Factory 预处理检查。V2.2：对缺口题补生成，盲审实际视图后追加，不改动已冻结版本 | `assemble_A/v2_merge/`：`prepare.py` → `summarize.py` → `export_release.py` → `check_export.py`、`lf_preprocess_check.py`；补生成 `sft_v2_topup.py`、`sft_v2_topup_k3_followon.py`；`assemble_A/v2_2_topup/`：`prepare_review.py` → `finalize.py` |
| 6 构造库 B（插值） | 在库 A 同题上让教师按 v6 协议重做，把它某一步的裁图强制换成偏框，要求从真实回执里发现并纠正；早期标定 → v3r2 → v4 → v4c（与库 A 条件对齐）。生成顺序 GPT-5.6 Sol → GPT-6.1 Sol 补缺 → GPT-6 Astra 补缺 | `interp_recovery_*.py`、`interp_sol_*.py`、`interp_v3r2_*.py`、`interp_v4_*.py`、`interp_v4r2_campaign_20261013.py`、`interp_v4c_sol56_20261013.py` → `interp_v4c_sol61_fill_20261013.py` → `interp_v4c_astra_fill_20261013.py`、`interp_v4c_pace_20261013.py` |
| 7 选取与交付库 B | 判定哪条 episode 算正确、哪些题需要下一个生成器补做；装配两库并导出 | `interp_v4c_select_20261013.py`、`build_interp_v4c_delivery_20261013.py` |

之后：按轮导出 SFT 数据见 `../sft/`。

## 已知问题（2026-10-18 独立审查）

对库 A 的审查（11 条定向视觉复核：8 条保留，3 条需修订）指出：
- 有"答对了但 caption 有错"的轨迹。例如描述了不存在的曲线交叉，或换了目标对象却没有说明。
- 筛选时，凡有一次裁图被审核器标为 `no_new_evidence` 或 `uncertain`，整条轨迹就进不了 core_ready（`assemble_A/v2_merge/summarize.py` 第 97–116 行）。合理的排除式搜索可能因此被误删。
- 教师看到的初始图有 3,443 张超过学生的像素预算（589,824），学生处理后是否还看得清，尚未核对。

**接下来可能会重新构建轨迹里的裁图部分。方案已有，确定后会更新到本仓库。**
