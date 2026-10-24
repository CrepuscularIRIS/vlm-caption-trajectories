# 80 个审阅记录：轨迹与逐题标注

先读[进展报告](../../docs/archive/30_rl_credit_report_20261022.md)，再按病例查看。

s00–s39 检验获取新细节；u00–u39 检验利用已有证据。两批可能复用同一题/轨迹，不能当作 80 个独立样本。

本目录公开学生完整可见 caption、裁图动作、独立标注、13 个分段锚点及审阅图片。按用户追加要求补图；题库完整原文仍不分发。

每例按初始图→caption/动作→回执排列。利用证据批次有129张视图引用；焦点回执批次有147张（含T/C），共276处引用、274个独立PNG。图片按SHA-256去重保存，与原审阅文件逐字节一致。

s批次只包含焦点裁图及之前的视图和T/C，不含后续未审阅回执；u批次包含完整轨迹各次回执。C是诊断对照，不是学生旧轨迹中的输入。图片来自记录所注明的数据源，项目代码许可不改变原图的来源权利。

`annotation_snapshot.json` 保留标注完成时状态，其中 probe_executed=false 是历史快照；其后的 v1 测量见 logs/mve/segval_v1。

| 病例 | 审阅结论/类型 | 诊断分层 |
|---|---|---|
| [s00](receipt/s00.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s01](receipt/s01.md) | unknown | 配对回执门槛未过 |
| [s02](receipt/s02.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s03](receipt/s03.md) | redundant | 配对回执门槛未过 |
| [s04](receipt/s04.md) | redundant | 配对回执门槛未过 |
| [s05](receipt/s05.md) | redundant | 配对回执门槛未过 |
| [s06](receipt/s06.md) | redundant | 配对回执门槛未过 |
| [s07](receipt/s07.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s08](receipt/s08.md) | redundant | 配对回执门槛未过 |
| [s09](receipt/s09.md) | redundant | 配对回执门槛未过 |
| [s10](receipt/s10.md) | necessary | 配对回执门槛未过 |
| [s11](receipt/s11.md) | redundant | 配对回执门槛未过 |
| [s12](receipt/s12.md) | unknown | 配对回执门槛未过 |
| [s13](receipt/s13.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s14](receipt/s14.md) | redundant | 配对回执门槛未过 |
| [s15](receipt/s15.md) | redundant | 配对回执门槛未过 |
| [s16](receipt/s16.md) | redundant | 配对回执门槛未过 |
| [s17](receipt/s17.md) | unknown | 配对回执门槛未过 |
| [s18](receipt/s18.md) | redundant | 配对回执门槛未过 |
| [s19](receipt/s19.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s20](receipt/s20.md) | redundant | 配对回执门槛未过 |
| [s21](receipt/s21.md) | unknown | 配对回执门槛未过 |
| [s22](receipt/s22.md) | redundant | 配对回执门槛未过 |
| [s23](receipt/s23.md) | evidence_not_used | 配对回执门槛未过 |
| [s24](receipt/s24.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s25](receipt/s25.md) | unknown | 配对回执门槛未过 |
| [s26](receipt/s26.md) | unknown | 配对回执门槛未过 |
| [s27](receipt/s27.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s28](receipt/s28.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s29](receipt/s29.md) | redundant | 配对回执门槛未过 |
| [s30](receipt/s30.md) | redundant | 配对回执门槛未过 |
| [s31](receipt/s31.md) | redundant | 配对回执门槛未过 |
| [s32](receipt/s32.md) | unknown | 配对回执门槛未过 |
| [s33](receipt/s33.md) | redundant | 配对回执门槛未过 |
| [s34](receipt/s34.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s35](receipt/s35.md) | unknown | 配对回执门槛未过 |
| [s36](receipt/s36.md) | redundant | 配对回执门槛未过 |
| [s37](receipt/s37.md) | unknown | 配对回执门槛未过 |
| [s38](receipt/s38.md) | no_evidence_or_misleading | 配对回执门槛未过 |
| [s39](receipt/s39.md) | redundant | 配对回执门槛未过 |
| [u00](utilization/u00.md) | rule_false_positive | 未入选分段诊断 |
| [u01](utilization/u01.md) | option_mapping | core |
| [u02](utilization/u02.md) | rule_false_positive | 未入选分段诊断 |
| [u03](utilization/u03.md) | row_column_binding | 未入选分段诊断 |
| [u04](utilization/u04.md) | option_mapping | core |
| [u05](utilization/u05.md) | option_mapping | 未入选分段诊断 |
| [u06](utilization/u06.md) | option_mapping | core |
| [u07](utilization/u07.md) | rule_false_positive | 未入选分段诊断 |
| [u08](utilization/u08.md) | misread_existing_evidence | 未入选分段诊断 |
| [u09](utilization/u09.md) | rule_false_positive | 未入选分段诊断 |
| [u10](utilization/u10.md) | other | 未入选分段诊断 |
| [u11](utilization/u11.md) | option_mapping | core |
| [u12](utilization/u12.md) | misread_existing_evidence | mixed |
| [u13](utilization/u13.md) | misread_existing_evidence | 未入选分段诊断 |
| [u14](utilization/u14.md) | goal_drift | mixed |
| [u15](utilization/u15.md) | misread_existing_evidence | 未入选分段诊断 |
| [u16](utilization/u16.md) | rule_false_positive | 未入选分段诊断 |
| [u17](utilization/u17.md) | option_mapping | core |
| [u18](utilization/u18.md) | rule_false_positive | 未入选分段诊断 |
| [u19](utilization/u19.md) | option_mapping | 未入选分段诊断 |
| [u20](utilization/u20.md) | rule_false_positive | 未入选分段诊断 |
| [u21](utilization/u21.md) | row_column_binding | 未入选分段诊断 |
| [u22](utilization/u22.md) | rule_false_positive | 未入选分段诊断 |
| [u23](utilization/u23.md) | misread_existing_evidence | 未入选分段诊断 |
| [u24](utilization/u24.md) | rule_false_positive | 未入选分段诊断 |
| [u25](utilization/u25.md) | goal_drift | 未入选分段诊断 |
| [u26](utilization/u26.md) | misread_existing_evidence | 未入选分段诊断 |
| [u27](utilization/u27.md) | hallucinated_receipt | 未入选分段诊断 |
| [u28](utilization/u28.md) | row_column_binding | core |
| [u29](utilization/u29.md) | rule_false_positive | 未入选分段诊断 |
| [u30](utilization/u30.md) | goal_drift | 未入选分段诊断 |
| [u31](utilization/u31.md) | other | 未入选分段诊断 |
| [u32](utilization/u32.md) | rule_false_positive | 未入选分段诊断 |
| [u33](utilization/u33.md) | row_column_binding | mixed |
| [u34](utilization/u34.md) | goal_drift | 未入选分段诊断 |
| [u35](utilization/u35.md) | misread_existing_evidence | 未入选分段诊断 |
| [u36](utilization/u36.md) | misread_existing_evidence | mixed |
| [u37](utilization/u37.md) | goal_drift | core |
| [u38](utilization/u38.md) | row_column_binding | core |
| [u39](utilization/u39.md) | option_mapping | core |
