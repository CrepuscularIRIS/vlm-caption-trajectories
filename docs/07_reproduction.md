# 代码阅读与复现

## 公开仓库可直接运行的检查

Python 3.10+，无需第三方包：

```bash
python scripts/verify_snapshot.py
python scripts/inspect_case.py cases/mme_realworld_lite_22254__glm.json
```

`verify_snapshot.py` 从已发布元数据重新计算 30 题 / 67 记录 / 16 导出 / 1-0-0 覆盖，核对导出与既有 LF 报告行、案例时序及文件哈希。请求计数来自本地原记录汇总；公开包不含原始 API 回执，因此无法仅凭该包重新数原始请求文件。

`inspect_case.py` 使用发布的真实协议解析器，检查一个案例各步的格式、行为、引用时序与软风险。它不会把通过格式检查的 WEAK 案例变成训练正例。图片没有分发，两个脚本都不认证像素事实。

CI 运行上述离线检查，并检查全部 5 个案例的协议。不会加载模型、训练或调用外部生成服务。

## 实际源码快照

这些文件复制自当前上游实现。对应源路径与 SHA-256 见 [source_hashes.json](../data/source_hashes.json)。这是审阅时的代码快照，不证明所有历史请求都使用了完全相同的源码字节。注释可能滞后于函数逻辑，例如审计器旧模块说明只写风险/裁图排序，而现函数还优先稀缺行为。理解行为以函数实现与本次记录为准。

|文件|阅读入口|可运行边界|
|---|---|---|
|[caption_v51.py](../code/reference/caption_v51.py)|SYSTEM、task_hints、coord_status、check_step、to_sharegpt|标准库 + 已含 caption_v3；可离线解析|
|[caption_v3.py](../code/reference/caption_v3.py)|parse_step、标签适配、基础检查|v5.1 复用的解析依赖；其旧 SYSTEM 不是本批运行提示|
|[run_caption_v51.py](../code/reference/run_caption_v51.py)|call_turn、rollout|需要原项目 API/图像环境适配器及 Pillow|
|[audit_v51.py](../code/reference/audit_v51.py)|candidates、audit_candidate、check_fact|需要原项目 API 适配器、图像和冻结轨迹|
|[select_v51.py](../code/reference/select_v51.py)|behaviors、recovery_like、main|需要项目 ROOT、审计 sidecar、人工表与完整图像路径|
|[corpus_scoring.py](../code/reference/corpus_scoring.py)|score|标准库确定性答案规范化|
|[lf_preprocess_check.py](../code/reference/lf_preprocess_check.py)|main|原固定 LLaMA-Factory 环境与本地 processor 路径；公开 CI 不执行|

生成/审计运行器引用的 `probe_api`、`trajgen_models`、`run_minio3_replica` 等项目适配模块未分发。这里是**实现阅读快照与可复算结果包**，不是已经移植好的端到端生成 SDK。模型凭证、供应商配置与本地模型文件也不包含在内。

## 上游验证记录

此前在原项目执行：

```bash
PYTHONPATH=obs/src python -m pytest -q \
  obs/src/test_caption_v51.py obs/src/test_audit_v51.py obs/src/test_corpus_v1.py
```

50 passed，已有 unit 标记警告。这是上游测试记录，不是上述测试文件已随公开仓库完整发布的声明。87 项通过是执行方另外汇报的整体口径。

## 证据文件

|路径|内容|
|---|---|
|`data/snapshot.json`|67 条生成摘要、30 条选择记录、16 条导出元数据、LF 报告、补覆盖清单|
|`cases/*.json`|5 条真实可见短输出、动作、视图关系与哈希、人工记录|
|`data/source_hashes.json`|用于本地复算的源文件标识/哈希及公开代码来源|
|`data/expected_summary.json`|当前冻结的数量与行为覆盖预期，由检查脚本独立复算比较|
|`manifest.sha256.json`|本次发布文件内容哈希，便于检查意外改动|

哈希证明文件一致性，不证明内容真实或科学结论。没有分发的原文件不能仅靠哈希恢复。
