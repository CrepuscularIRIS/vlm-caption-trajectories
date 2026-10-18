# 构造侧代码脱敏记录

代码从本机 `obs/src/`（生成、审核、筛选、库 B 插值）和库 A 两次装配目录按原样复制，只做下表的**字面路径替换**（长串优先，按出现次数全部替换），其他字节不变。
替换后的 `${...}` 是占位符，Python 不会自动展开：运行前需把它们换成本机路径（例如 `sed`）。

## 路径映射

| 原路径 | 占位符 |
|---|---|
| 本机项目根目录 | `${ICML_DIR}` |
| 项目根目录下的 `.env`（凭证，运行时读取，不在代码中） | `${ICML_DIR}/.env` |
| 本地模型代理目录及其 `.env`（LiteLLM / CLIProxy） | `${PROXY_DIR}` |
| conda Python | `${PYTHON}` |
| conda bin | `${CONDA_BIN}` |
| Claude Code 包装入口 | `${CC_BIN}` |
| Gemini agy CLI | `${AGY_BIN}` |
| 用户 bin | `${LOCAL_BIN}` |
| 用户主目录 | `${HOME}` |
| 私有数据仓库 | `${DATA_DIR}/vlm-sft-v2-2` |
| 模型权重 | `${MODEL_DIR}` |
| Python 环境 | `${ENV_DIR}` |
| HF 缓存 | `${HF_HOME}` |
| CLI 工作目录 | `${WORK_DIR}` |
| 基准数据 | `${BENCH_DIR}` |

其余说明：
- 代码里的 `http://127.0.0.1:<port>` 是本机模型代理（LiteLLM / CLIProxy）的本地地址，保留原样；API key 一律在运行时从 `.env` 读取，代码与收录文件中没有凭证（已用正则扫描 `sk-`/`ghp_`/`Bearer`/长随机串）。
- `src/minio3_prompts.py` 的示例取自 Mini-o3 论文附录 E–F（arXiv 2509.07969）的公开提示，不含本项目评测题。
- `assemble_A/v2_merge/export_release.py` 原本把主目录前缀替换为 `[private-home]/`，该字面串同样被换成 `${HOME}/`。

## 逐文件

| 文件 | 原始 sha256（前 16 位） | 当前 sha256（前 16 位） | 替换 |
|---|---|---|---|
| `assemble_A/v2_2_topup/finalize.py` | `da54918e10e70818` | `25f7c0e044e401e1` | <原路径>→${ICML_DIR} ×1 |
| `assemble_A/v2_2_topup/lf_preprocess_check.py` | `ae53fa957480aaf2` | `869c12876baa44bd` | <原路径>→${MODEL_DIR} ×1；<原路径>→${ENV_DIR} ×1 |
| `assemble_A/v2_2_topup/prepare_review.py` | `140f0eec98b3731f` | `0ed7b83039916caa` | <原路径>→${ICML_DIR} ×1 |
| `assemble_A/v2_2_topup/verify_cached_mask.py` | `eff4eefdaf5c5292` | `5596a46e04fdb9d0` | <原路径>→${MODEL_DIR} ×1 |
| `assemble_A/v2_merge/check_export.py` | `024620db825e4c12` | `6874301af16259ae` | <原路径>→${ICML_DIR} ×1 |
| `assemble_A/v2_merge/export_release.py` | `5e379bd3228fa7e7` | `a955a16ccd1f28ad` | <原路径>→${ICML_DIR} ×1；<原路径>→${HOME}/ ×1 |
| `assemble_A/v2_merge/lf_preprocess_check.py` | `ae53fa957480aaf2` | `869c12876baa44bd` | <原路径>→${MODEL_DIR} ×1；<原路径>→${ENV_DIR} ×1 |
| `assemble_A/v2_merge/prepare.py` | `9d8f9e71e4cf1561` | `d10d70e90fc2d51c` | <原路径>→${ICML_DIR} ×1 |
| `assemble_A/v2_merge/summarize.py` | `63d41ada45aa96ca` | `63d41ada45aa96ca` | 无 |
| `assemble_A/v2_merge/verify_cached_mask.py` | `091f531847f3807c` | `fa14ff66b7ddafcb` | <原路径>→${MODEL_DIR} ×1 |
| `src/agy_reader.py` | `b7506ed5ce19793d` | `e525809e67d0fa21` | <原路径>→${AGY_BIN} ×1；<原路径>→${WORK_DIR} ×1 |
| `src/api_reader.py` | `9081a6902b53e5da` | `9081a6902b53e5da` | 无 |
| `src/audit_tiered.py` | `f94419040b28420f` | `f94419040b28420f` | 无 |
| `src/audit_v51.py` | `b22692ab98d0149d` | `b22692ab98d0149d` | 无 |
| `src/behavior_prompts.py` | `0a4f6a5f04e51fe3` | `0a4f6a5f04e51fe3` | 无 |
| `src/build_interp_v4c_delivery_20261013.py` | `5ea1ee395bf71d07` | `a1dcb706c44cbc58` | <原路径>→${DATA_DIR}/vlm-sft-v2-2 ×1 |
| `src/build_selection_4k.py` | `4243a3f86381c1df` | `5765be935d981310` | <原路径>→${ICML_DIR} ×1 |
| `src/build_selection_multistep_first.py` | `3fb0bb4f74d4b7cd` | `438e902f98a6c4f9` | <原路径>→${ICML_DIR} ×1 |
| `src/caption_generation_v2.py` | `876fa914c49dc853` | `876fa914c49dc853` | 无 |
| `src/caption_tiered.py` | `575228829d96f03b` | `575228829d96f03b` | 无 |
| `src/caption_v3.py` | `9b89f09914030acd` | `9b89f09914030acd` | 无 |
| `src/caption_v51.py` | `af0a4a7b6b92813f` | `af0a4a7b6b92813f` | 无 |
| `src/caption_v6.py` | `4901017ab36ebacd` | `4901017ab36ebacd` | 无 |
| `src/cc_proxy_reviewer.py` | `59a37e39e671f1be` | `4ec14832205e1e68` | <原路径>→${CC_BIN} ×2；<原路径>→${WORK_DIR} ×1 |
| `src/common.py` | `abc6382b2cbbb64a` | `b157fd32f30f2f04` | <原路径>→${ICML_DIR} ×1 |
| `src/consensus_pipeline_step.py` | `b6b2636a03567381` | `4bb743dae17b44e1` | <原路径>→${ENV_DIR} ×2 |
| `src/corpus_scoring.py` | `73b162c4c8eb1e43` | `73b162c4c8eb1e43` | 无 |
| `src/cursor_anthropic.py` | `3f8c2a9228e7acdc` | `c5dbed5b45d69c8c` | <原路径>→${PROXY_DIR}/.env ×1 |
| `src/export_protocol_check.py` | `b3b9f246d8dfb4cc` | `b3b9f246d8dfb4cc` | 无 |
| `src/export_selection_4k.py` | `9eaec0f677ae00b5` | `8d93df89b74bad37` | <原路径>→${ICML_DIR} ×1 |
| `src/funnel_audit.py` | `7756ecbca06d64b6` | `7756ecbca06d64b6` | 无 |
| `src/funnel_gen.py` | `95f63d54856d8ea3` | `95f63d54856d8ea3` | 无 |
| `src/funnel_gen_extra.py` | `67b9834b4a0518d3` | `67b9834b4a0518d3` | 无 |
| `src/funnel_pool.py` | `a98d46c274df017b` | `a98d46c274df017b` | 无 |
| `src/funnel_recover_views.py` | `6037fa34d54289de` | `6037fa34d54289de` | 无 |
| `src/funnel_select.py` | `c3079abc94e55016` | `c3079abc94e55016` | 无 |
| `src/funnel_stats_report.py` | `8b2eed2c15977f9b` | `8b2eed2c15977f9b` | 无 |
| `src/funnel_supplement_queue.py` | `f3783cbdabf857c0` | `f3783cbdabf857c0` | 无 |
| `src/glm_direct.py` | `10301426ea9b795d` | `51d26091c1b21bad` | <原路径>→${PROXY_DIR}/.env ×1 |
| `src/interp_recovery_calibration.py` | `8a0b7e4173f7286f` | `8a0b7e4173f7286f` | 无 |
| `src/interp_recovery_full.py` | `b043003569a8a4c1` | `b043003569a8a4c1` | 无 |
| `src/interp_recovery_full_r2.py` | `0225a29c9adb5140` | `0225a29c9adb5140` | 无 |
| `src/interp_recovery_v2.py` | `e137f6ca87910866` | `e137f6ca87910866` | 无 |
| `src/interp_sol_full_4040.py` | `15110137a72449b0` | `e014f3a181652ae4` | <原路径>→${PROXY_DIR}/litellm ×1；<原路径>→${PYTHON} ×1 |
| `src/interp_sol_full_campaign.py` | `906d47745386f7a1` | `6cec86ec246f7776` | <原路径>→${PYTHON} ×1 |
| `src/interp_sol_pro_continue_20261012.py` | `cc54bbd21f417669` | `77012ace749272bd` | <原路径>→${PYTHON} ×1 |
| `src/interp_sol_pro_retry_backoff_20261012.py` | `ea37fa9143d5e484` | `ea37fa9143d5e484` | 无 |
| `src/interp_sol_pro_retry_dedup_20261012.py` | `73bca0e0517025a8` | `73bca0e0517025a8` | 无 |
| `src/interp_v3r2_campaign_20261012.py` | `6441e69b4a8853cf` | `788107552e1accac` | <原路径>→${PYTHON} ×1 |
| `src/interp_v3r2_plan_20261012.py` | `d77d1db93752668b` | `d77d1db93752668b` | 无 |
| `src/interp_v4_campaign_20261013.py` | `fc1d10c8046afb0b` | `fc1d10c8046afb0b` | 无 |
| `src/interp_v4_plan_20261013.py` | `e88ff3f0264ba5c6` | `e88ff3f0264ba5c6` | 无 |
| `src/interp_v4c_astra_20261013.py` | `eff2032f09513861` | `eff2032f09513861` | 无 |
| `src/interp_v4c_astra_fill_20261013.py` | `94a5e9c35bdacc4e` | `94a5e9c35bdacc4e` | 无 |
| `src/interp_v4c_pace_20261013.py` | `be71eb34bbc1a399` | `be71eb34bbc1a399` | 无 |
| `src/interp_v4c_select_20261013.py` | `21cceb3da5d77d7d` | `21cceb3da5d77d7d` | 无 |
| `src/interp_v4c_sol56_20261013.py` | `c26270fb0c554255` | `c26270fb0c554255` | 无 |
| `src/interp_v4c_sol61_20261013.py` | `ae694cb69ad062d0` | `ae694cb69ad062d0` | 无 |
| `src/interp_v4c_sol61_fill_20261013.py` | `ca07851d353f1e67` | `ca07851d353f1e67` | 无 |
| `src/interp_v4r2_campaign_20261013.py` | `7e66e655b7a639fa` | `7e66e655b7a639fa` | 无 |
| `src/local_vlm.py` | `2d27295a0622b95a` | `577299daf0326eb1` | <原路径>→${HF_HOME} ×1 |
| `src/minio3_prompts.py` | `805b1bfbbd3aab26` | `805b1bfbbd3aab26` | 无 |
| `src/mme_rw_reextract.py` | `63cf42dc8bb9f217` | `494158ee6893b928` | <原路径>→${HF_HOME} ×1；<原路径>→${BENCH_DIR} ×1 |
| `src/models_v51.py` | `eb12014cf2dfb1ae` | `eb12014cf2dfb1ae` | 无 |
| `src/monitor_step.py` | `15ea1465a4376a43` | `15ea1465a4376a43` | 无 |
| `src/probe_api.py` | `93aa2f0b8654b2c7` | `93aa2f0b8654b2c7` | 无 |
| `src/probe_replica_api.py` | `e08a4c29bf308e14` | `e08a4c29bf308e14` | 无 |
| `src/reconcile_corpus_splits.py` | `67c99518ca104147` | `67c99518ca104147` | 无 |
| `src/rescore_open.py` | `1ad20722604f5233` | `1ad20722604f5233` | 无 |
| `src/run_api_t0.py` | `1454dac057aba33a` | `1454dac057aba33a` | 无 |
| `src/run_caption_v51.py` | `b878ef0238b1f1f3` | `b878ef0238b1f1f3` | 无 |
| `src/run_minio3_replica.py` | `b1c463b53727d43e` | `69973dcc1e52406e` | <原路径>→${ICML_DIR} ×1；<原路径>→${HF_HOME} ×1 |
| `src/scorer_audit.py` | `ece396d4147e3bc0` | `4b22b818f9105679` | <原路径>→${ICML_DIR} ×1 |
| `src/scoring.py` | `cfa0cb4b85bbbfd4` | `cfa0cb4b85bbbfd4` | 无 |
| `src/select_tiered.py` | `cf2af991ef00a41e` | `cf2af991ef00a41e` | 无 |
| `src/sft_v2_topup.py` | `c8200c82704cb2d3` | `c8200c82704cb2d3` | 无 |
| `src/sft_v2_topup_k3_followon.py` | `9a3a678e5a6b1f30` | `277287ecb135db24` | <原路径>→${ICML_DIR} ×1 |
| `src/tiered_transport.py` | `261f6c320356d310` | `32fabed7b850bb3b` | <原路径>→${WORK_DIR} ×1 |
| `src/trajgen_models.py` | `4ef386f3cebe70fc` | `0604e6b27b57a7be` | <原路径>→${PROXY_DIR}/.env ×1；<原路径>→${ICML_DIR}/.env ×1；<原路径>→${MODEL_DIR} ×1 |
