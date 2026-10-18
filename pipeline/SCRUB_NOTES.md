# 脱敏记录（SCRUB_NOTES）

代码从训练服务器按原样复制（`cat` 读取），复制后逐文件核对 sha256 与服务器一致；之后只做下表所列的路径替换，其他代码不变。
替换由一次性脚本按“精确字符串 + 预期次数”执行，次数不符即中止。代码中标 `[scrub]` 的注释为替换时新增。

## 路径映射

| 原服务器路径 | 替换为 |
|---|---|
| `<server>/myproject`（代码工作目录） | `${PROJECT_DIR}`；插件目录 `grpo_v51`、`grpo_v6`、`grpo_v51_cropcost`、`grpo_v51_cropfloor` 映射为 `${PROJECT_DIR}/grpo/v51` 等 |
| `<server>/myproject/outputs` | `${OUTPUT_DIR}` |
| `<server>/datasets` | `${DATA_DIR}` |
| `<server>/models` | `${MODEL_DIR}` |
| `<server>/cache/hf` | `${CACHE_DIR}/hf` |
| EasyR1 检出目录 | `${EASYR1_DIR}` |
| EasyR1 / LLaMA-Factory Python 环境 | `${EASYR1_ENV}` / `${LF_ENV}` |
| 协议参考实现目录（`caption_v51.py`、`caption_v3.py`） | `CAPTION_REF_DIR`，默认本仓库 `code/reference` |
| v4 构造目录（`caption_v6.py`） | `CAPTION_PROTOCOL_DIR`，默认本仓库 `pipeline/protocol` |

`/tmp/ray_*`（Ray 临时目录）为通用路径，保留不改。主机名、端口、ssh 命令在收录的代码中不存在。

## 行为上的变化（相对原代码）

- `grpo/*/caption_env.py`：协议目录由硬编码改为环境变量，默认指向本仓库内对应文件；新增 `import os`。
- `grpo/v51_cropcost/caption_contracts.py`：`ROLLOUT_LOG` 改为 `${OUTPUT_DIR:-outputs}/grpo_v3_A_cropcost_rollouts.jsonl`。
- `sft/build_grpo_data.py`：`CAPTION_GRPO_CODE` 原有默认值（服务器项目目录）去掉，现为必需；`DATA_DIR` 必需。
- `sft/export_v22_turnwise.py`、`sft/build_v4_sft.py`、`eval/eval_v4.py`、`eval/grill_checks.py`：`DATA_DIR` / `EASYR1_DIR` / `PROJECT_DIR` 必需（未设置即 KeyError）。
- `eval/grill_checks.py`：分析模块目录由硬编码改为脚本所在目录。
- SFT yaml 注释中的启动命令原写 `sft_vlm_v2_2.yaml`（复制旧配置留下的注释），改为 `${PROJECT_DIR}/sft/<this_file>.yaml`；两个 yaml 头部注释“Database I v4 插值版”同为旧注释，未改。
- 未改动：`sft/export_lf_turnwise.py`（库 B 的按轮导出，服务器 `vlm-sft-v2-2/scripts/` 原文件，路径均为相对仓库根）、`grpo/easyr1_602a128_local.diff`（服务器 EasyR1 检出上 `git diff` 的原文）。

## 逐项替换

| 文件 | 次数 | 原文（首行，服务器前缀记为 `<server>`） | 替换为（首行） |
|---|---|---|---|
| `grpo/v51/caption_env.py` | 1x | `import re` | `import os  # [scrub]` |
| `grpo/v51/caption_env.py` | 1x | `REF = Path("<server>/datasets/vlm-caption-trajectories/code/reference")` | `_REPO = Path(__file__).resolve().parents[3]  # [scrub] repo root (pipeline/grpo/<variant>/ -> repo)` |
| `grpo/v51_cropcost/caption_env.py` | 1x | `import re` | `import os  # [scrub]` |
| `grpo/v51_cropcost/caption_env.py` | 1x | `REF = Path("<server>/datasets/vlm-caption-trajectories/code/reference")` | `_REPO = Path(__file__).resolve().parents[3]  # [scrub] repo root (pipeline/grpo/<variant>/ -> repo)` |
| `grpo/v51_cropfloor/caption_env.py` | 1x | `import re` | `import os  # [scrub]` |
| `grpo/v51_cropfloor/caption_env.py` | 1x | `REF = Path("<server>/datasets/vlm-caption-trajectories/code/reference")` | `_REPO = Path(__file__).resolve().parents[3]  # [scrub] repo root (pipeline/grpo/<variant>/ -> repo)` |
| `grpo/v6/caption_env.py` | 1x | `import re` | `import os  # [scrub]` |
| `grpo/v6/caption_env.py` | 1x | `REF = Path("<server>/datasets/vlm-sft-v2-2/construction_v4")` | `_REPO = Path(__file__).resolve().parents[3]  # [scrub] repo root (pipeline/grpo/<variant>/ -> repo)` |
| `grpo/v51_cropcost/caption_contracts.py` | 1x | `ROLLOUT_LOG = "<server>/myproject/outputs/grpo_v3_A_cropcost_rollouts.jsonl` | `ROLLOUT_LOG = __import__("os").path.join(__import__("os").environ.get("OUTPUT_DIR", "outputs"), "grpo_v3_A_cro` |
| `grpo/scripts/grpo_2plus2.sh` | 1x | `<server>/cache/hf` | `${CACHE_DIR}/hf` |
| `grpo/scripts/grpo_2plus2.sh` | 1x | `cd <easyr1>` | `cd ${EASYR1_DIR}` |
| `grpo/scripts/grpo_2plus2.sh` | 1x | `<server>/envs/easyr1/bin` | `${EASYR1_ENV}/bin` |
| `grpo/scripts/grpo_2plus2.sh` | 1x | `<server>/myproject/grpo_v51` | `${PROJECT_DIR}/grpo/v51` |
| `grpo/scripts/grpo_2plus2.sh` | 1x | `<server>/myproject/grpo_v6` | `${PROJECT_DIR}/grpo/v6` |
| `grpo/scripts/grpo_2plus2.sh` | 4x | `<server>/myproject/outputs` | `${OUTPUT_DIR}` |
| `grpo/scripts/grpo_2plus2.sh` | 2x | `<server>/datasets` | `${DATA_DIR}` |
| `grpo/scripts/grpo_cropcost.sh` | 1x | `<server>/cache/hf` | `${CACHE_DIR}/hf` |
| `grpo/scripts/grpo_cropcost.sh` | 1x | `cd <easyr1>` | `cd ${EASYR1_DIR}` |
| `grpo/scripts/grpo_cropcost.sh` | 1x | `<server>/envs/easyr1/bin` | `${EASYR1_ENV}/bin` |
| `grpo/scripts/grpo_cropcost.sh` | 1x | `<server>/myproject/grpo_v51_cropcost` | `${PROJECT_DIR}/grpo/v51_cropcost` |
| `grpo/scripts/grpo_cropcost.sh` | 2x | `<server>/myproject/outputs` | `${OUTPUT_DIR}` |
| `grpo/scripts/grpo_cropcost.sh` | 1x | `<server>/datasets` | `${DATA_DIR}` |
| `grpo/scripts/grpo_cropfloor.sh` | 1x | `<server>/cache/hf` | `${CACHE_DIR}/hf` |
| `grpo/scripts/grpo_cropfloor.sh` | 1x | `cd <easyr1>` | `cd ${EASYR1_DIR}` |
| `grpo/scripts/grpo_cropfloor.sh` | 1x | `<server>/envs/easyr1/bin` | `${EASYR1_ENV}/bin` |
| `grpo/scripts/grpo_cropfloor.sh` | 1x | `<server>/myproject/grpo_v51_cropfloor` | `${PROJECT_DIR}/grpo/v51_cropfloor` |
| `grpo/scripts/grpo_cropfloor.sh` | 3x | `<server>/myproject/outputs` | `${OUTPUT_DIR}` |
| `grpo/scripts/grpo_cropfloor.sh` | 1x | `<server>/datasets` | `${DATA_DIR}` |
| `grpo/scripts/grpo_caption.sh` | 1x | `<server>/cache/hf` | `${CACHE_DIR}/hf` |
| `grpo/scripts/grpo_caption.sh` | 1x | `cd <easyr1>` | `cd ${EASYR1_DIR}` |
| `grpo/scripts/grpo_caption.sh` | 1x | `<server>/envs/easyr1/bin` | `${EASYR1_ENV}/bin` |
| `grpo/scripts/grpo_caption.sh` | 2x | `<server>/myproject/outputs` | `${OUTPUT_DIR}` |
| `grpo/scripts/grpo_caption.sh` | 1x | `<server>/myproject/easyr1_caption.py` | `${PROJECT_DIR}/easyr1_caption.py` |
| `grpo/scripts/grpo_caption.sh` | 1x | `<server>/myproject/reward_caption.py` | `${PROJECT_DIR}/reward_caption.py` |
| `grpo/scripts/grpo_caption.sh` | 2x | `<server>/datasets` | `${DATA_DIR}` |
| `sft/sft_v22_turnwise.yaml` | 1x | `cd <server>/datasets/vlm-sft-v2-2` | `cd ${DATA_DIR}/vlm-sft-v2-2` |
| `sft/sft_v22_turnwise.yaml` | 1x | `PATH=<lf_env>/bin:$PATH` | `PATH=${LF_ENV}/bin:$PATH` |
| `sft/sft_v22_turnwise.yaml` | 1x | `<server>/myproject/sft_vlm_v2_2.yaml` | `${PROJECT_DIR}/sft/<this_file>.yaml` |
| `sft/sft_v22_turnwise.yaml` | 1x | `model_name_or_path: <server>/models/Qwen3-VL-4B-Instruct` | `model_name_or_path: ${MODEL_DIR}/Qwen3-VL-4B-Instruct  # [scrub] LLaMA-Factory does not expand env vars; subst` |
| `sft/sft_v22_turnwise.yaml` | 1x | `dataset_dir: <server>/datasets/vlm-sft-v2-2/data` | `dataset_dir: ${DATA_DIR}/vlm-sft-v2-2/data` |
| `sft/sft_v22_turnwise.yaml` | 1x | `output_dir: <server>/myproject/outputs/` | `output_dir: ${OUTPUT_DIR}/` |
| `sft/sft_v4c.yaml` | 1x | `cd <server>/datasets/vlm-sft-v2-2` | `cd ${DATA_DIR}/vlm-sft-v2-2` |
| `sft/sft_v4c.yaml` | 1x | `PATH=<lf_env>/bin:$PATH` | `PATH=${LF_ENV}/bin:$PATH` |
| `sft/sft_v4c.yaml` | 1x | `<server>/myproject/sft_vlm_v2_2.yaml` | `${PROJECT_DIR}/sft/<this_file>.yaml` |
| `sft/sft_v4c.yaml` | 1x | `model_name_or_path: <server>/models/Qwen3-VL-4B-Instruct` | `model_name_or_path: ${MODEL_DIR}/Qwen3-VL-4B-Instruct  # [scrub] LLaMA-Factory does not expand env vars; subst` |
| `sft/sft_v4c.yaml` | 1x | `dataset_dir: <server>/datasets/vlm-sft-v2-2/data` | `dataset_dir: ${DATA_DIR}/vlm-sft-v2-2/data` |
| `sft/sft_v4c.yaml` | 1x | `output_dir: <server>/myproject/outputs/` | `output_dir: ${OUTPUT_DIR}/` |
| `sft/export_v22_turnwise.py` | 1x | `import json, re` | `import json, os, re  # [scrub] os` |
| `sft/export_v22_turnwise.py` | 2x | `<server>/datasets/vlm-sft-v2-2/data/` | `os.environ["DATA_DIR"] + "/vlm-sft-v2-2/data/` |
| `sft/build_v4_sft.py` | 1x | `import json` | `import json` |
| `sft/build_v4_sft.py` | 2x | `<server>/datasets/vlm-sft-v2-2/data/` | `os.environ["DATA_DIR"] + "/vlm-sft-v2-2/data/` |
| `sft/build_grpo_data.py` | 1x | `输出: <server>/datasets/grpo_caption/` | `输出: ${DATA_DIR}/grpo_caption/` |
| `sft/build_grpo_data.py` | 1x | `.get("CAPTION_GRPO_CODE", "<server>/myproject")` | `["CAPTION_GRPO_CODE"]` |
| `sft/build_grpo_data.py` | 1x | `SFT = Path("<server>/datasets/vlm-sft-v2-2")` | `SFT = Path(__import__("os").environ["DATA_DIR"]) / "vlm-sft-v2-2"  # [scrub]` |
| `sft/build_grpo_data.py` | 1x | `B = Path("<server>/datasets/benchmarks")` | `B = Path(__import__("os").environ["DATA_DIR"]) / "benchmarks"  # [scrub]` |
| `sft/build_grpo_data.py` | 1x | `GRPO_POOL_DIR", "<server>/datasets/grpo_caption` | `GRPO_POOL_DIR", _os.environ["DATA_DIR"] + "/grpo_caption` |
| `eval/eval_v4.py` | 1x | `--code-dir <server>/myproject/grpo_v51\|grpo_v6` | `--code-dir ${PROJECT_DIR}/grpo/v51\|v6` |
| `eval/eval_v4.py` | 1x | `sys.path.insert(0, "<easyr1>")` | `sys.path.insert(0, os.environ["EASYR1_DIR"])  # [scrub]` |
| `eval/eval_v4.py` | 1x | `pool = a.pool or ("<server>/datasets/grpo_caption_v51_v3" if "v51" in a.code_dir` | `pool = a.pool or (os.environ["DATA_DIR"] + "/grpo_caption_v51_v3" if "v51" in a.code_dir  # [scrub]` |
| `eval/grill_checks.py` | 1x | `sys.path.insert(0, "<server>/myproject/cropfloor_analysis")` | `sys.path.insert(0, str(Path(__file__).resolve().parent))  # [scrub]` |
| `eval/grill_checks.py` | 1x | `import collections, json, re, sys` | `import collections, json, os, re, sys  # [scrub] os` |
| `eval/grill_checks.py` | 1x | `P = Path("<server>/myproject")` | `P = Path(os.environ["PROJECT_DIR"])  # [scrub] expects outputs/ and evidence/ under it` |
| `eval/grill_checks.py` | 1x | `pq.read_table("<server>/datasets/grpo_caption_v51_v3/val.parquet")` | `pq.read_table(os.environ["DATA_DIR"] + "/grpo_caption_v51_v3/val.parquet")` |
| `eval/README.zh.md` | 1x | `cd <server>/myproject` | `cd ${PROJECT_DIR}   # 原服务器项目目录；输出在 outputs/、evidence/ 下` |
| `eval/README.zh.md` | 2x | `--code-dir <server>/myproject/grpo_v51` | `--code-dir ${PROJECT_DIR}/grpo/v51` |
| `eval/README.zh.md` | 1x | `--code-dir <server>/myproject/grpo_v6` | `--code-dir ${PROJECT_DIR}/grpo/v6` |

## 文件来源与原始 sha256

“原始”为复制后、脱敏前的值，均与服务器上的 `sha256sum` 一致。`protocol/caption_v6.py` 取自本地副本，与服务器 GRPO B 实际导入的副本一致；`eval/` 中的 `eval_v4.py`、`analyze_v4.py`、`traj_*.py`、测试和 `README.zh.md` 取自本地评测目录，其中 `analyze_v4.py`、`traj_metrics.py`、`traj_utils.py` 与服务器分析目录副本一致。

| 文件 | 原始 sha256（前 16 位） | 当前 sha256（前 16 位） | 已脱敏 |
|---|---|---|---|
| `eval/README.zh.md` | `fcb627a04a545247` | `67401ce50d640474` | 是 |
| `eval/analyze_v4.py` | `bf893ed65569be62` | `bf893ed65569be62` | 否 |
| `eval/compare_cropcost.py` | `21dc76d935b0dfbd` | `21dc76d935b0dfbd` | 否 |
| `eval/eval_v4.py` | `8261f7b03131b811` | `79056369a7531016` | 是 |
| `eval/grill_checks.py` | `4b136fa5fa9fed46` | `112605ee63466586` | 是 |
| `eval/group_signal.py` | `1653e6ad75c415a6` | `1653e6ad75c415a6` | 否 |
| `eval/test_traj_metrics.py` | `e6eb0eb00470b18f` | `e6eb0eb00470b18f` | 否 |
| `eval/test_traj_utils.py` | `57d467d80064ac59` | `57d467d80064ac59` | 否 |
| `eval/traj_metrics.py` | `d53f5a6f612eb9c5` | `d53f5a6f612eb9c5` | 否 |
| `eval/traj_utils.py` | `88b54d95b3ef1fbf` | `88b54d95b3ef1fbf` | 否 |
| `grpo/easyr1_602a128_local.diff` | `6c41c076d99f17a7` | `6c41c076d99f17a7` | 否 |
| `grpo/scripts/grpo_2plus2.sh` | `813da4414c0a3957` | `d36868642d2baad8` | 是 |
| `grpo/scripts/grpo_caption.sh` | `d0d8c0291a2e8668` | `b85f72c114b101f1` | 是 |
| `grpo/scripts/grpo_cropcost.sh` | `b405c881b36d1957` | `1318eace150aab79` | 是 |
| `grpo/scripts/grpo_cropfloor.sh` | `19a44ab9ee760255` | `c5c482f50fa38106` | 是 |
| `grpo/v51/caption_contracts.py` | `134baf985e4ce8d8` | `134baf985e4ce8d8` | 否 |
| `grpo/v51/caption_env.py` | `6ec561a25003c48b` | `3281dbe878f848d3` | 是 |
| `grpo/v51/easyr1_caption.py` | `4416324f1d69e0d5` | `4416324f1d69e0d5` | 否 |
| `grpo/v51/reward_caption.py` | `d58f2e19c4194590` | `d58f2e19c4194590` | 否 |
| `grpo/v51_cropcost/caption_contracts.py` | `566a165f0d7fe007` | `52b02d967a47ab76` | 是 |
| `grpo/v51_cropcost/caption_env.py` | `6ec561a25003c48b` | `3281dbe878f848d3` | 是 |
| `grpo/v51_cropcost/easyr1_caption.py` | `e6278921aef3a11d` | `e6278921aef3a11d` | 否 |
| `grpo/v51_cropcost/reward_caption.py` | `95dcd5dafedd9a05` | `95dcd5dafedd9a05` | 否 |
| `grpo/v51_cropcost/test_reward_cropcost.py` | `5f83fc04662e2295` | `5f83fc04662e2295` | 否 |
| `grpo/v51_cropfloor/caption_contracts.py` | `ffd141bc1da8b3b4` | `ffd141bc1da8b3b4` | 否 |
| `grpo/v51_cropfloor/caption_env.py` | `6ec561a25003c48b` | `3281dbe878f848d3` | 是 |
| `grpo/v51_cropfloor/easyr1_caption.py` | `822794f34381b3fb` | `822794f34381b3fb` | 否 |
| `grpo/v51_cropfloor/reward_caption.py` | `8fc390fb4a8f07b4` | `8fc390fb4a8f07b4` | 否 |
| `grpo/v51_cropfloor/test_cropfloor.py` | `13bec0faee6c6f24` | `13bec0faee6c6f24` | 否 |
| `grpo/v6/caption_contracts.py` | `134baf985e4ce8d8` | `134baf985e4ce8d8` | 否 |
| `grpo/v6/caption_env.py` | `2373280b0f53b77e` | `f6bdc4149561e421` | 是 |
| `grpo/v6/easyr1_caption.py` | `61641c686086c276` | `61641c686086c276` | 否 |
| `grpo/v6/reward_caption.py` | `d58f2e19c4194590` | `d58f2e19c4194590` | 否 |
| `protocol/caption_v6.py` | `4901017ab36ebacd` | `4901017ab36ebacd` | 否 |
| `sft/build_grpo_data.py` | `63cd901a96c4261c` | `ded339f59b20634f` | 是 |
| `sft/build_v4_sft.py` | `d45a8b8f3ff5480e` | `0eb3be4b5a00cd6b` | 是 |
| `sft/export_lf_turnwise.py` | `db537eeb075ef81f` | `db537eeb075ef81f` | 否 |
| `sft/export_v22_turnwise.py` | `9de2a2468adf7572` | `3859e9f15731ed36` | 是 |
| `sft/sft_v22_turnwise.yaml` | `94877588ca69c66e` | `752e10822b78a100` | 是 |
| `sft/sft_v4c.yaml` | `7f0410e6561da815` | `a5aae44f818beabd` | 是 |

协议参考实现未复制，引用仓库已有文件：`code/reference/caption_v51.py`（`af0a4a7b6b92813f`）、`code/reference/caption_v3.py`（`9b89f09914030acd`），与服务器 GRPO 导入目录中的副本 sha256 一致。

## 2026-10-18 补充：`grpo/v51_cc/` 与 `grpo/scripts/grpo_cc.sh`

- `v51_cc/` 是在服务器 `grpo_v51_cropcost` 上新写的插件，原样收录：新增 `counter_credit.py`、`test_counter_credit.py`、`calibrate_eps.py`；改动 `easyr1_caption.py`（记录裁图前状态、对答对轨迹跑探针、安装优势补丁）、`caption_contracts.py`（rollout 日志路径改为环境变量 `CAPTION_ROLLOUT_LOG`，日志加价格字段）、`reward_caption.py`（奖励只看正确与否）。以上文件不含任何路径。
- `v51_cc/caption_env.py` 与 `v51_cropcost/caption_env.py` 相同（同一份脱敏版本）。
- `scripts/grpo_cc.sh` 按上表同样的映射替换路径（cache/hf、EasyR1 目录、插件目录、outputs、datasets、EasyR1 环境）。

## 2026-10-18 晚补充：统计脚本与 KL / 对照接续脚本

- `v51_cc/group_stats.py`、`v51_cc/log_metrics.py`：新写的离线统计脚本，原样收录，不含路径（在服务器 `outputs/` 目录下按相对文件名运行）。
- `scripts/grpo_kl.sh`：由服务器 `grpo_cropfloor.sh` 改出（λ 固定为 0、开 KL、run 名与 rollout 日志改为 `grpo_v3_A_kl_*`），路径按上表映射替换。
- `scripts/wait_then_kl.sh`、`scripts/wait_then_ctrl.sh`：服务器上的接续脚本，只把项目目录替换为 `${PROJECT_DIR}`。
