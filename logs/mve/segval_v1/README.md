# 第一次分段价值诊断：逐次结果记录

对应[进展报告](../../../docs/archive/30_rl_credit_report_20261022.md)第 3 节问题 4、Open Scientist 第016–018步。13例×4前缀×16次=832行，0次参数更新。原实验执行者运行，本次发布只读取和复算。

- `runs_sft.jsonl`：7例（5 core + 2 mixed），448行。
- `runs_r1.jsonl`：6例（4 core + 2 mixed），384行。
- 历史汇总：[segval_summary.json](../segval_summary.json)。

每行的 `id/group/policy/prefix/j` 标识一次测量；`y`为终答是否正确，`crops`包含前缀已有裁图，`r=y*(1-.01*crops)`，`gen_tokens`为新增生成token数，`partial_chars`为注入的助手前缀字符数。解析失败、超长或没有终答在原代码中记0，没有另存逐类停止原因。`j`是该前缀的采样序号，不是跨前缀相同随机种子；实际种子规则见测量代码。

**这是逐次分数日志，不是完整生成轨迹。** v1实现没有保存832次续写文本或prompt token IDs，不能从这里恢复。原来用于看图的80条审阅记录与本次832次续跑是不同材料，见[data/review_20261022](../../../data/review_20261022/README.md)。

CPU复核命令见[代码入口](../../../pipeline/eval/mve/README.md)。

本次发布逐字节核对了服务器原文件与本地副本，SHA-256 一致：

| 文件 | SHA-256 |
|---|---|
| runs_sft.jsonl | `942e66295878d03727fb61da448a0cd421dc652f90993be9cbf2c1d9b73e4696` |
| runs_r1.jsonl | `ef3648aa38a44ba8ce8ad75f6079c54c95f6f3028f77d8f7b6d6df018a9c032d` |

运行源为原实验的 `evidence/segval_20261022/run/`；不由此推断 v1 前缀构造或因果解释已获验证。
