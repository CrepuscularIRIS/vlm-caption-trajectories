# 生产流水线阅读快照

来自本地 `obs/src/` 的 2026-10-04 公开整理。历史 `../` 文件仍保留，不把新旧实现悄悄混为同一版本。哈希与摘录方式见 [来源清单](../../../data/production_provenance_20261004.json)。

版本边界：`funnel_audit.py` 在统计冻结之后增加了可选 Sonnet CLI 路由，因此其公开源哈希与 10-03 审计记录不同；两种哈希均已保存。其余五个源文件哈希与冻结审计记录一致。本页没有验证或宣称新路由已经运行，统计页的实际路由仍以旧批次回执为准。

|文件|负责什么|
|---|---|
|[funnel_gen.py](funnel_gen.py)|按材料和稳定散列选完整 demo，真实逐轮 crop，目标字段白名单，格式修复与缓存|
|[teacher_context_excerpt.py](teacher_context_excerpt.py)|仅摘录 GUIDANCE 与 TeacherContext；demo 视图命名空间与目标轨迹分离|
|[funnel_select.py](funnel_select.py)|题级答案/家族漏斗、CPU 门、候选排序；不证明语义质量|
|[funnel_audit.py](funnel_audit.py)|Sonnet 过程审查，独立盲读与比较，预算/传输/分歧状态|
|[audit_tiered.py](audit_tiered.py)|前缀视图约束、结构字段检查、CPU 检查和基础独立审核|
|[build_audit_pools.py](build_audit_pools.py)|将已有审核记录分到候选、拒绝、争议、待核等池|

这些是供审阅的实际代码快照，**不是可直接启动生产任务的独立安装包**。`models_v51`、`tiered_transport`、`agy_reader`、`api_reader`、题图和生产配置等依赖未分发；导入路径也沿用源工程。公开脚本不要求运行这些生成器。

已知问题保留原貌，详见 [质量缺口](../../../docs/09_production_review_20261004.md)：fallback_unreviewed 仍可能自动通过；终答引用视图包可能缺原图；坐标检测存在特定漏检。发布不等于这些问题已修复。
