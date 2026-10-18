# eval_v4：导出四个模型的完整轨迹（本机假模型全流程已通过，GPU 未跑）

| 文件 | 内容 |
|---|---|
| `eval_v4.py` | 导出脚本，替代 `eval_v3.py` 做行为分析 |
| `traj_utils.py` | 种子、IoU、扰动框、改写动作框，不依赖 GPU |
| `traj_metrics.py`、`analyze_v4.py` | 本机分析：E1/E2 指标与聚类 bootstrap（云端不用跑） |
| `test_traj_utils.py`、`test_traj_metrics.py` | 10 个单元测试，本机已通过 |

方案与指标见上一级 `V3_REFLECTION_AND_NEXT_EVAL_20261016.zh.md` 第 5 节；审计结论见上一级 `V3_AUDIT_20261016.zh.md`。

## 和 eval_v3.py 的区别

| | eval_v3 | eval_v4 |
|---|---|---|
| 保存内容 | 每题答对次数 | 每个采样的每一轮：原文、解析字段、协议检查错误、绝对框、token 数、结束原因 |
| 种子 | 无 | 每个（题、采样、轮次）一个固定种子 |
| 异常 | `except: pass` 吞掉 | 写进记录的 `error` 字段 |
| 打分 | 只看最后一轮 | `ok` 同 eval_v3；`ok_train_rule` 同训练（任一轮违反协议或被截断即 0 分） |
| 图像最小像素 | 262,144（与训练不一致） | 1,024，与 SFT、RL 一致；`--min-px 262144` 可复现 eval_v3 |
| 速度 | 逐个采样串行 | 32 题 × 4 采样同轮批量生成 |
| 断点续跑 | 无 | 按（uid, sample_idx）跳过已完成的 |
| 扰动 | 无 | `--perturb first_crop` |
| 头部记录 | 无 | vLLM/torch 版本、题库和 `caption_env.py` 的 sha256、模型 config 的 sha256 |

**扰动怎么做**：模型第一次在原图上裁框 b 时，环境改裁一个明显不同的框 b′（绝对 IoU < 0.1），并把 b′ 写回这一轮 `<grounding>` 里的坐标，模型的其他文字一律不动。这样历史看起来就像模型自己框错了，回执里报告的区域和动作里的坐标一致，与 v4c 库中“强插偏框”的样子相同；模型能发现错误的唯一线索是新视图的内容。相同（uid, 采样）且首框相同时，A 和 B 得到同一个 b′。

## 云端运行

四张卡都空着，每张卡跑一个模型。

```
cd ${PROJECT_DIR}   # 原服务器项目目录；输出在 outputs/、evidence/ 下
E=<eval_v4 目录>

# 0) 单元测试
python -m pytest -q $E                                       # 应为 10 passed

# 1) 冒烟：A_sft，20 题，跑两遍，看是否可复现
CUDA_VISIBLE_DEVICES=0 python $E/eval_v4.py --model outputs/sft_v22_turnwise_merged \
    --code-dir ${PROJECT_DIR}/grpo/v51 --tag A_sft --out-dir evidence/v3_behavior_smoke --limit 20
CUDA_VISIBLE_DEVICES=0 python $E/eval_v4.py --model outputs/sft_v22_turnwise_merged \
    --code-dir ${PROJECT_DIR}/grpo/v51 --tag A_sft --out-dir evidence/v3_behavior_smoke2 --limit 20
CUDA_VISIBLE_DEVICES=0 python $E/eval_v4.py --model outputs/sft_v4c_merged \
    --code-dir ${PROJECT_DIR}/grpo/v6 --tag B_sft --out-dir evidence/v3_behavior_smoke \
    --limit 20 --perturb first_crop

# 2) 正式 E1：四个模型各占一张卡，同时跑
#    tag    model                                                   code-dir
#    A_sft  outputs/sft_v22_turnwise_merged                         grpo_v51
#    A_rl   outputs/grpo_v3_A/global_step_160/actor/huggingface     grpo_v51
#    B_sft  outputs/sft_v4c_merged                                  grpo_v6
#    B_rl   outputs/grpo_v3_B/global_step_160/actor/huggingface     grpo_v6
#    输出目录 evidence/v3_behavior，日志 evidence/v3_behavior/log_<tag>.txt

# 3) E2：同样四个模型，加 --perturb first_crop，输出同一目录（文件名带 _perturb）
```

## 冒烟时核对

1. 每条记录都有 `turns`、`boxes_abs`、`n_crops`、`gen_tokens`、`ok_train_rule`；`error` 非空的不超过 1%。
2. 两遍冒烟的 `turns[*].text` 逐条比较，报告完全相同的比例。vLLM 批次内的数值不保证逐位一致，低于 90% 时把差异样例贴回来，不要改代码。
3. 抽 3 条：`turns[i].box_abs` 等于 `boxes_abs[i+1]`；`boxes_abs[0]` 是整图。
4. 扰动冒烟：`perturb.iou_abs < 0.1` 的比例；`turns[0].text_in_history` 里的坐标等于 `perturb.shown_bbox`。
5. 冒烟只有 80 个采样，准确率不和 V3 比。正式跑完后，总体准确率与 `V3_COMPARISON.md` 相差超过 2.5 个点时先报告，再分析。图像最小像素改了，有小幅差异属正常。

## 交回什么

`evidence/v3_behavior/` 整个目录：每个模型的 `traj_*.jsonl`、`header_*.json`、日志，加上实际耗时。本机读回后算 E1、E2 的指标和聚类 bootstrap。

## 已知局限

- 本机用假模型（假 vLLM、假处理器、真实的 `caption_env.py`、`caption_v6.py`、`reward_caption.py`）跑通了导出、扰动、改写动作框和断点续跑；真实模型的输出格式、显存和速度要在云端冒烟时确认。
- 只扰动第一次、且来源为原图的裁图；首轮直接作答的采样不受扰动，单独统计其比例。
- 边长超过图像一半的大框可能找不到 IoU < 0.1 的平移，记录 `perturb.ok=false`，分析时剔除。
