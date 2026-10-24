# 2026-10-23 设置审计的复算数据

主文：[docs/37](../../docs/37_baselines_and_scale_20261023.md)。本目录是已有文件的 CPU 聚合与新计划；不是新训练结果。

- `gradient_audit.json`：12 份旧 CSV，保留来源与 SHA-256；范数为裁剪前值，系数为 .1 阈值的公式推算。
- `existing_eval_audit.json`：只读聚合 63 份旧评测 JSONL，保留 SHA-256 和 UID 集摘要；不含正在运行的 confirm_20261024 部分评测。
- `historical_tradeoff_data.json`：图中 18 个历史条件的 33 个 E1/E2 点；每个均核对 552 题×4 采样、相同 UID 集、首裁+后续=总 crop。
- `next_recipe.json`：新一轮共同配方规格；数据 manifest/revision 尚需执行准备阶段填入，不是可直接运行的 EasyR1 配置。

成本：`total_logical_tokens = prompt_tokens + gen_tokens`，prompt_tokens 是各次调用实际 expanded input 长度之和，包括图像和重送历史。它不是 cache-adjusted prefill FLOPs，也不是训练 actor 的 token 总数。

正确率：历史图使用 `ok` 以对齐已发布历史表，另外保留 `ok_train_rule`；缺失数据不填零。按题平均四采样，训练 seed 不因 bootstrap 或评测采样而增加。E2 的无工具基线不适用，64题 DL 不混入本图。图中未画置信区间，不据点估计声称非劣。

在仓库根目录重绘：

```bash
python pipeline/eval/plot_setting_audit.py
```

输出为 `media/setting_audit_20261023/` 下 PNG/SVG。此命令不联网、不调用模型、不训练。
