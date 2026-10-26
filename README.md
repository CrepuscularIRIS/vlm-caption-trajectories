# VLM Caption Trajectories

Qwen3-VL-4B 学习“先写简短证据 Caption，再按需裁图”的视觉推理：SFT 教会这个行为，RL 再优化准确率与观测成本之间的权衡。

**状态（2026-10-24）：Probe 阶段，方法还没有定型。** 实验计划以导师要求的全量逐条对账为准（导师原话逐行、逐条对照、训练配置与任务缺口），之前的全部文档与记录已归档。

## 现在读

| 文件 | 内容 |
|---|---|
| **[docs/01 项目总览](docs/01_overview.md)** | 我们在做什么、设计、代码位置、实验矩阵、当前运行、接下来的实验思路 |
| [docs/02 实验细节与已知问题](docs/02_known_issues.md) | 一：docs/01 所列发现的条件、数据范围与完整计数；二：代码审查发现的问题、影响、修复状态 |

## 目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `pipeline/` | 轨迹构造、SFT、GRPO、评测代码（路径已脱敏，见 `pipeline/SCRUB_NOTES.md`） | 在用；`pipeline/eval/baselines/` 是新的原模型评测 |
| `trajectories/` | 库 A（4,040）/ 库 B（3,983）SFT 轨迹，去题干与图像 | 在用（A 是 SFT 起点） |
| `logs/`、`media/` | 训练/评测日志与图 | 历史结果，保留可复算 |
| `data/`、`cases/`、`web/`、`scripts/`、`code/`、`prompts/` | V1–V2.2 轨迹发布、审核表、案例、网页目录、校验脚本 | 历史发布，CI 仍校验 |
| `docs/archive/` | 2026-10-24 之前的全部旧报告、变更记录与旧 README（合并索引在 `docs/archive/README.md`） | **只作记录**，不再作为依据 |

旧文档、旧变更记录与旧 README 的更新记录都在 [docs/archive/README.md](docs/archive/README.md)。

## 不能从本仓库得出的结论

- 历史 RL 结果来自小规模、`max_grad_norm=0.1` 的设置，并且 PPO clip 实际从未生效。这些结果只能作为探索记录，不能当作方法有效或无效的证据。
- 内部评测集 E1 是开发与比较用的集合，不是独立测试集。8k RL 题池和 A-SFT 都接触过部分官方 MMMU-Pro / V* 题，详见 [docs/02](docs/02_known_issues.md)。

校验：`python scripts/verify_snapshot.py`，以及 `.github/workflows/verify.yml` 中的其余脚本。
