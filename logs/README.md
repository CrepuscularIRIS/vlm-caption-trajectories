# 训练日志与评测汇总

只收录指标、配置和汇总文本；不含题面、图像路径、模型输出、rollout 逐条记录或 checkpoint。路径已替换为 `${DATA_DIR}`、`${OUTPUT_DIR}`、`${PROJECT_DIR}`、`${EASYR1_DIR}`、`${MODEL_DIR}`（含义见 [../pipeline/README.md](../pipeline/README.md) 第 5 节）。

## 1. GRPO run 一览

时间为服务器本地时间（2026 年）。E1 = 552 题 × 4 采样（T=0.7，同种子）的总体准确率。

| run | 库 / SFT 起点 | 奖励 | 优势 | 训练步数 | GPU | 起止 | E1 |
|---|---|---|---|---|---|---|---|
| `grpo_v3_A` | A（V2.2，v5.1）/ `sft_v22_turnwise` | `correct` | GRPO std 归一 | 165 | 2（0,1） | 10-15 19:15 → 10-16 03:47 | A_rl@160 **.664**（A_sft .552） |
| `grpo_v3_B` | B（v4c 插值，v6）/ `sft_v4c` | `correct` | GRPO std 归一 | 165 | 2（2,3） | 10-15 19:22 → 10-16 03:48 | B_rl@160 **.638**（B_sft .539） |
| `grpo_v3_A_cropcost` | A | `correct × (1 − 0.01·crops)` | GRPO std 归一 | 56（手动停止） | 2（0,1） | 10-17 14:12 → 15:53 | cropcost@40 **.511** |
| `grpo_v3_A_floor_l025` | A | `correct × (1 − 0.025·crops)` | (r−mean)/max(std, 0.25) | 165 | 2 | 10-17 16:44 → 23:41 | fl025 **.649** |
| `grpo_v3_A_floor_l050` | A | `correct × (1 − 0.05·crops)` | (r−mean)/max(std, 0.25) | 165 | 2 | 10-17 18:10 → 10-18 00:53 | fl050 **.647** |
| `grpo_v3_A_floor_s43_l000` | A，seed 43 | `correct`（λ=0） | (r−mean)/max(std, 0.25) | 3（停止） | 2 | 10-18 10:51 → 11:02 | 无 |
| `grpo_v3_A_floor_s43_l050` | A，seed 43 | `correct × (1 − 0.05·crops)` | (r−mean)/max(std, 0.25) | 3（停止） | 2 | 10-18 10:51 → 11:02 | 无 |
| `grpo_v3_A_cc`（卡 A） | A | `correct`，价格通道另计 | GRPO + (P − 组均值) | 165 | 2（0,1） | 10-18 17:04 → 10-19 03:25 | A_cc160 **.666** |
| `grpo_v3_A_cc_dsasym`（卡 B） | A | 同卡 A | 同卡 A；DS 只丢全错组 | 165 | 2（2,3） | 10-18 17:04 → 10-19 06:58 | A_ccb160 **.656** |
| `grpo_v3_A_kl_k1e2` | A | `correct` | GRPO + KL（low_var_kl 0.01） | 165 | 2（0,1） | 10-19 04:41 → 14:17 | A_kl160 **.653** |
| `grpo_v3_AO_base` | 答案式 SFT `sft_v22_answeronly` | `correct` | GRPO | 165 | 2（2,3） | 10-19 07:58 → 10:59 | AO_rl160 **.492**（AO_sft .458） |
| `grpo_v3_A_floor_ctrl_l000` | A | `correct`（λ=0） | (r−mean)/max(std, 0.25)（混合组与原版相同） | 165 | 2（2,3） | 10-19 10:59 → 18:49 | 对照重跑，用于分组统计（未做 E1） |
| `grpo_v3_A_floor_orig_s42` | A | `correct`（λ=0） | (r−mean)/max(std, 0.25)（与原版等价） | 165 | 2（2,3） | 10-20 | 原版对照，每 40 步存权重；E1 见 docs/29 |
| `grpo_v3_A_dv_s42` | A | `correct` | 同上 + 首轮 token 加 s·dV | 165 | 2（0,1） | 10-20 | 首轮价值差；第 160 步 **.664**（原版 .648），见 docs/29 |

- 两个 seed 43 run 位于服务器 `outputs/stopped_seed43_20261018/`，是人为停止的复跑；日志里完整打印了第 1–3 步的指标，第 4 步生成中被停止。
- 卡 A / 卡 B 的 `_metrics.csv` 由 `pipeline/grpo/v51_cc/log_metrics.py` 从控制台日志导出（训练中导出，非最终；含 `cc/price_mean`、`cc/verified`、`cc/crops`）。rollout 分组统计见 `grpo/group_stats_20261018.md`。
- 卡 A / 卡 B 之前的所有 run：动态采样关闭（`online_filtering=false`）；clip 0.2/0.3 为默认开启，但 `on_policy_reuse_log_probs=true` 下重要性比恒为 1，`pg_clipfrac_*` 与 `ppo_kl` 全程为 0；无 KL；无格式奖励。

## 2. 文件

### `grpo/`

| 文件 | 内容 |
|---|---|
| `<run>_metrics.csv` | 每步一行。step 0 只有起点验证（`val/*`）。列见下 |
| `<run>_config.json` | EasyR1 启动时打印的完整解析配置（JSON），外加 `_caption_plugin_from_log`：日志里记录的插件目录、协议模块，以及 crop-floor 的 `std_floor` / `crop_lambda` |

`<run>` 为上表中的 7 个 run 名。CSV 列：

- EasyR1 每步打印的标量，原名保留，层级用 `/` 连接：`actor/*`（entropy_loss、grad_norm、lr、pg_loss、pg_clipfrac_*、ppo_kl）、`critic/{advantages,returns,rewards,score}/{max,mean,min}`、`reward/accuracy`（训练批答对率）、`reward/overall`（训练批奖励均值，含裁图代价）、`response_length/*`、`prompt_length/*`、`global_seqlen/*`、`perf/*`、`timing_s/*`、`timing_per_token_ms/*`；验证步还有 `val/accuracy_reward`（验证集 440 题答对率）、`val/overall_reward`、`val_response_length/*`、`val_prompt_length/*`。
- `rollout_log/*`：插件每个 rollout worker 打印的 `[caption] trajectories=… terminal_answers=… invalid=… length=… context_limits=…`，按训练步求和（每步 12 题 × 16 = 192 条；验证批的行已排除）。
- `floor/*`（仅 crop-floor run）：每步 `[floor]` 行，即组数、std 被下限替换的组数、这些组中最大的 |advantage|。
- `rollout_jsonl/*`（仅 cropcost 与 crop-floor run）：在服务器上从逐条 rollout 记录**只读汇总**得到的每步统计，未复制原始记录：`n_rollouts`、`mean_crops`（平均裁图次数）、`direct_rate`（未裁图直接作答的比例）、`correct_mean`、`score_mean`、`terminal_answer_rate`、`gen_tokens_mean`（该条记录的 response_mask token 数）、`direct_correct_rate`。A/B 主实验没有逐条记录，故无这些列。

### `sft/`

| 目录 | run | 内容 |
|---|---|---|
| `sft_v22_turnwise_lora/` | 库 A 的 SFT（LLaMA-Factory LoRA，2 epoch，284 步，10-15） | `trainer_log.jsonl`（每 5 步 loss/lr）、`trainer_state.json`（含 log_history 与 epoch 末 eval_loss）、`all_results.json`、`train_results.json`、`eval_results.json`、`adapter_config.json`（基座路径已替换为 `${MODEL_DIR}`） |
| `sft_v4c_lora/` | 库 B 的 SFT（同设置，406 步，10-14） | 同上 |

最终 eval_loss：A 0.744，B 0.548（两库验证切分不同，不可直接比较）。

### `eval/`

| 文件 | 对应 run | 内容 |
|---|---|---|
| `analyze_out_20261016.txt` | A_sft、A_rl（`grpo_v3_A`@160）、B_sft、B_rl（`grpo_v3_B`@160） | 双库 E1/E2 完整输出：总体与分层指标、95% cluster bootstrap 对比、E2 扰动结果 |
| `E1_CROPCOST_20261017.md` | `grpo_v3_A_cropcost`@40 | 第一轮裁图代价的 E1 汇总与停跑原因 |
| `compare_round2_20261018.txt` | `grpo_v3_A_floor_l025`、`_l050` 对比 A_rl、A_sft | 第二轮 E1/E2：分层准确率、直答率、重复裁图、bootstrap 对比 |
| `grill_checks_20261018.txt` | 同上 | 复核检查：日志 val 值、E2 配对增量、重复裁图、训练组构成、逐题 Δacc 与 Δcrops |

上述文本由 `../pipeline/eval/` 中的 `analyze_v4.py`、`compare_cropcost.py`、`grill_checks.py` 生成，已核对不含题面或绝对路径。

### `twobranch/`

两分支测量与 dV GRPO 对比的汇总（[docs/29](../docs/archive/29_twobranch_dv_grpo_20261020.md)）：`per_question_v2.csv`（逐题两分支正确率、裁图数、生成 token、未满足首轮约束次数）、`test_v2.json`（分类与置换检验）、`test_v1_unmet_scored_wrong.json`（作废的第一版）、`inspection_verdicts.csv`（人工抽看判定，只含 uid / 类别 / 判定）、`dv_table.json`、`dv_summary.json`、`compare.json`（两组 E1/E2 各步与训练中按 dV 分组的首轮裁图比例）。

### `sec6/`

第 6 节汇总（原 docs/30，现并入[进展报告](../docs/archive/30_rl_credit_report_20261022.md)）：`turn_decomp.json`（按轮次拆解）、`probe_summary.json`（固定上下文探针）、`reader_summary.json`（冻结读者门槛）、`four_case.json`（裁图前后读者对错的四种情况）。

### `fc/`、`fcz/`、`dl/`（10-22/23，见 docs/30 第 7 节）

| 文件 | 内容 |
|---|---|
| `fc/compare_fc.json` | 四种情况信用 vs r1/r0：E1/E2 第 80/160 步的准确率、裁图、直接作答，配对 bootstrap；训练窗口与 `[fc_credit]`/`[fc_forks]` 汇总 |
| `fcz/compare_fcz.json` | 零信用 vs 四种情况（及 r1）：同上；预注册判定 |
| `dl/compare_dl.json` | 首动作监督 vs β=0 对照：U_choice（逐状态 p 与 δ，按 uid）、clean/E2/恢复、训练日志摘要、预注册判定 |
