# 训练管线：数据构造 → SFT → GRPO → 评测

本目录收录 caption + 裁图工具（crop tool）多轮 VLM 的训练与评测代码，基座模型为 Qwen3-VL-4B-Instruct。
代码按服务器上实际运行的版本收录，只做了路径脱敏（见 [SCRUB_NOTES.md](SCRUB_NOTES.md)）。
训练指标、配置和评测汇总见 [../logs/](../logs/README.md)。

## 1. 流程

| 阶段 | 做什么 | 代码 |
|---|---|---|
| 构造 | 题池、多教师真实裁图生成、审核、筛选、库 A/B 装配 | [`construct/`](construct/README.zh.md) |
| 0 协议 | 系统提示、动作解析（`parse_step`）、回执文本（`receipt_text`）。库 A 用 v5.1，库 B 用 v6 | `../code/reference/caption_v51.py`、`../code/reference/caption_v3.py`（v51/v6 共同依赖，仓库已有，不重复收录）；`protocol/caption_v6.py` |
| 1 SFT 数据 | 轨迹库按"每个受监督的 assistant 轮一条样本"展开，配合 LLaMA-Factory `mask_history: true` 实现逐轮 loss 掩码 | 库 A：`sft/export_v22_turnwise.py`；库 B：`sft/export_lf_turnwise.py`（生成 `caption_sft_v4c_interp_train_turnwise.json`）；`sft/build_v4_sft.py` 是更早的合并式转换，未用于最终 B 的 SFT |
| 2 SFT | LLaMA-Factory LoRA（r=16, α=32, lr 1e-4, 2 epoch, cutoff 12288, `mask_history`, `preserve_thinking`），之后合并为 HF 模型 | `sft/sft_v22_turnwise.yaml`（A）、`sft/sft_v4c.yaml`（B） |
| 3 GRPO 题库 | 公开基准减去 SFT 已用题（题面去重），渲染与 SFT 同分布；train 2,000 / val 440 / V\* holdout 112 | `sft/build_grpo_data.py`（`CAPTION_GRPO_CODE` 决定协议版本） |
| 4 GRPO | EasyR1 `602a128` + 本项目多轮裁图插件：rollout 内环境执行裁图、回执拼回上下文，只对模型生成的 token 计 loss | `grpo/<variant>/`、`grpo/scripts/`、`grpo/easyr1_602a128_local.diff` |
| 5 评测 | E1：552 题（440 分层 val + 112 V\*）× 4 采样，T=0.7，固定种子，逐轮保存轨迹；E2：首次裁图被换成明显不同的框（IoU<0.1），看能否发现并纠偏 | `eval/eval_v4.py`（导出轨迹）、`eval/analyze_v4.py`、`eval/traj_metrics.py`、`eval/traj_utils.py`、`eval/compare_cropcost.py`、`eval/grill_checks.py`、`eval/group_signal.py`；说明见 `eval/README.zh.md` |

## 2. GRPO 插件目录

每个变体目录都是完整插件（`easyr1_caption.py` 入口 + `caption_env.py` 裁图环境 + `caption_contracts.py` 图像/位置/奖励契约 + `reward_caption.py`），通过 `CAPTION_GRPO_CODE` 选择：

| 目录 | 协议 | 奖励 | 优势 | 用于 |
|---|---|---|---|---|
| `grpo/v51/` | caption_v51 | `correct` | EasyR1 GRPO：(r−mean)/(std+1e-6) | `grpo_v3_A`（库 A） |
| `grpo/v6/` | caption_v6（裁图预算仍取 8） | `correct` | 同上 | `grpo_v3_B`（库 B） |
| `grpo/v51_cropcost/` | caption_v51 | `correct × (1 − 0.01 × crops)` | 同上 | `grpo_v3_A_cropcost`（第一轮裁图代价） |
| `grpo/v51_cropfloor/` | caption_v51 | `correct × (1 − λ × crops)`，λ 取自 `CAPTION_CROP_LAMBDA` | (r−mean)/max(std, `CAPTION_STD_FLOOR`=0.25) | `grpo_v3_A_floor_l025`（λ=0.025）、`_l050`（λ=0.05）、`stopped_seed43_20261018`（λ=0 / 0.05，seed 43） |

`v51` 与 `v6` 只差协议模块和 `caption_env.py` 的导入目录；`v51_cropcost` 在 `v51` 上加裁图计数与逐条 rollout 记录；`v51_cropfloor` 再加 std 下限优势。`test_reward_cropcost.py`、`test_cropfloor.py` 是 CPU 单测（`python test_*.py`，退出码 0 即通过）。

启动脚本（`grpo/scripts/`）：`grpo_2plus2.sh A|B`（A/B 主实验）、`grpo_cropcost.sh`、`grpo_cropfloor.sh <TAG> <LAMBDA> <GPUS>`。`grpo_caption.sh` 是更早的试跑脚本（KL 1e-3、n=8、调用 `${PROJECT_DIR}` 顶层的旧插件），未用于上述任何 run，仅作留档。

`grpo/easyr1_602a128_local.diff` 是 EasyR1 检出上的本地改动，复现必需：
- `verl/__init__.py`：在 driver 和每个 Ray worker 中按 `CAPTION_GRPO_CODE` 导入插件并调用 `install()`，未设置则报错；
- `ray_trainer.py` / `actor/config.py` / `dp_actor.py`：`worker.actor.on_policy_reuse_log_probs=true` 时跳过 old log-prob 重算，以当前前向的 detach 作为 old log-prob（要求 `ppo_epochs=1` 且 `global_batch_size == rollout_batch_size`）。

## 3. 奖励

- `correct`：轨迹必须以合法的终止动作结束（`caption_terminal == "answer"`），且模型生成文本中恰好一个 `<answer>X</answer>`，X 为 A–J 中的字母并等于标准答案，记 1；其他一律 0。
- 裁图变体只在答对时乘 `(1 − λ × crops)`，答错仍为 0；`crops` 是环境实际执行的裁图次数（预算 8）。
- 奖励只看模型生成的 token，不读环境回执；只写在最后一个生成 token 上。
- **没有任何格式奖励或标签奖励**：不奖励 `<answer>`/`<grounding>` 等标签、不奖励 hit/fallback 等字面标签、不奖励放弃作答、不奖励裁图本身，也不因未调用工具扣分。

## 4. 训练设置说明

- 所有 run 共同设置：rollout_batch_size=12，n=16，T=0.7，lr 1e-6（常数，warmup 5%），`kl_coef=0`（`disable_kl=true`），max_grad_norm 0.1，每轮最多 640 token，单题最多 8 次裁图，`trainer.max_steps=165`，2 卡/run，val 每 40 步。
- **所有 run 都关闭了动态采样**（`algorithm.online_filtering=false`，不做 DAPO 式按组过滤），全对/全错组都保留在批内。
- **PPO clip 用的是 EasyR1 默认值并处于开启状态**：`clip_ratio_low=0.2`、`clip_ratio_high=0.3`（dual-clip 3.0）。但因为 `on_policy_reuse_log_probs=true` 让 old log-prob 等于当前 log-prob，重要性比恒为 1，clip 实际不会触发；所有日志的 `pg_clipfrac_higher/lower` 和 `ppo_kl` 都是 0。

## 5. 环境变量

脱敏后的代码用以下变量代替原服务器绝对路径：

| 变量 | 含义 | 使用位置 |
|---|---|---|
| `PROJECT_DIR` | 代码工作目录；在本仓库中设为 `<repo>/pipeline`（插件在 `${PROJECT_DIR}/grpo/<variant>`）。`eval/grill_checks.py` 和 `eval/README.zh.md` 还假定其下有 `outputs/`、`evidence/`（原服务器布局） | 启动脚本、eval |
| `OUTPUT_DIR` | 训练产物：SFT LoRA/合并模型、GRPO checkpoint、rollout 记录 | 启动脚本、SFT yaml、`v51_cropcost` 的 rollout 日志（默认 `outputs`） |
| `DATA_DIR` | 数据根目录：`vlm-sft-v2-2/`（SFT 数据）、`benchmarks/`、`grpo_caption_v51_v3/`（A 题库）、`grpo_caption_v3/`（B 题库） | 数据构造、启动脚本、eval |
| `MODEL_DIR` | 基座模型目录（含 `Qwen3-VL-4B-Instruct`） | SFT yaml |
| `EASYR1_DIR` | 打过上述 diff 的 EasyR1 `602a128` 检出 | 启动脚本、`eval_v4.py` |
| `EASYR1_ENV` / `LF_ENV` | EasyR1 / LLaMA-Factory 的 Python 环境前缀 | 启动脚本 / SFT yaml 注释 |
| `CACHE_DIR` | HF 缓存（`HF_HOME=${CACHE_DIR}/hf`） | 启动脚本 |
| `CAPTION_GRPO_CODE` | 选择插件变体目录（必需） | EasyR1 hook、`build_grpo_data.py` |
| `CAPTION_REF_DIR` | `caption_v51.py`/`caption_v3.py` 所在目录，默认 `<repo>/code/reference` | `caption_env.py` |
| `CAPTION_PROTOCOL_DIR` | `caption_v6.py` 所在目录，默认 `<repo>/pipeline/protocol` | `grpo/v6/caption_env.py` |
| `CAPTION_CROP_LAMBDA`、`CAPTION_STD_FLOOR`、`CAPTION_ROLLOUT_LOG` | crop-floor 变体必需 | `grpo/v51_cropfloor/` |
| `GRPO_POOL_DIR` | 题库输出目录，默认 `${DATA_DIR}/grpo_caption` | `build_grpo_data.py` |

LLaMA-Factory 不展开 yaml 里的 `${...}`，使用 SFT 配置前需替换为实际路径。

## 6. 未收录

- 数据集、图像、checkpoint、parquet 题库、rollout jsonl（含题面与模型输出）。
- `caption_diagnostics`：插件仅在设置 `CAPTION_DIAG_DIR` 时导入，所有正式 run 均未设置，服务器上也无此文件。
