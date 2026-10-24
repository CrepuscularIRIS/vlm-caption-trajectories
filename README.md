# VLM Caption Trajectories

Qwen3-VL-4B 学习“先写简短证据 Caption，再按需裁图”的视觉推理：SFT 教会这个行为，RL 再优化准确率与观测成本之间的权衡。

**状态（2026-10-24）：Probe 阶段，方法还没有定型。** 导师 10-23 指出实验设置问题比方法本身更大。我们正在：
1. 修复已经定位的代码和统计问题；
2. 补齐 baseline 系列；
3. 用单 seed、更大规模重跑 RL。

## 现在读这三份

| 文件 | 内容 |
|---|---|
| [docs/40 已知问题与修复台账](docs/40_known_issues_20261024.md) | 代码审查发现的问题、影响、修复状态。**新结果要先过这一页的修复项** |
| [docs/39 Probe 协议（待导师确认）](docs/39_probe_protocol_for_mentor_20261024.md) | 共同 RL 配置、实验组、报告口径、需导师决定的问题 |
| [docs/38 导师要求逐条对账](docs/38_mentor_requirements_audit_20261024.md) | 10-14 → 10-23 每条要求做到了哪一步 |

[docs/37](docs/37_baselines_and_scale_20261023.md) 是 10-23 的 baseline 与规模计划（docs/39 在此基础上细化）。

## 目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `pipeline/` | 轨迹构造、SFT、GRPO、评测代码（路径已脱敏，见 `pipeline/SCRUB_NOTES.md`） | 在用；`pipeline/eval/baselines/` 是新的原模型评测 |
| `trajectories/` | 库 A（4,040）/ 库 B（3,983）SFT 轨迹，去题干与图像 | 在用（A 是 SFT 起点） |
| `logs/`、`media/` | 训练/评测日志与图 | 历史结果，保留可复算 |
| `data/`、`cases/`、`web/`、`scripts/`、`code/`、`prompts/` | V1–V2.2 轨迹发布、审核表、案例、网页目录、校验脚本 | 历史发布，CI 仍校验 |
| `docs/archive/` | 10-02 → 10-24 的全部旧报告（docs/01–36、CASEBOOK、Pro 审查等） | **只作记录**，结论以 docs/38–40 的更正为准 |

旧版 README 的全部更新记录在 [docs/archive/README_history_20261024.md](docs/archive/README_history_20261024.md)；旧文档编号与新路径的对照见 [docs/archive/README.md](docs/archive/README.md)。

## 不能从本仓库得出的结论

- 历史 RL 结果来自小规模、`max_grad_norm=0.1` 的设置，并且 PPO clip 实际从未生效。这些结果只能作为探索记录，不能当作方法有效或无效的证据。
- 内部评测集 E1 是开发与比较用的集合，不是独立测试集。8k RL 题池和 A-SFT 都接触过部分官方 MMMU-Pro / V* 题，详见 docs/40。

校验：`python scripts/verify_snapshot.py`，以及 `.github/workflows/verify.yml` 中的其余脚本。
