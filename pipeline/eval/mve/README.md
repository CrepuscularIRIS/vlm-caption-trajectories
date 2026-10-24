# MVE：问题、代码、可复核材料

先读[进展报告](../../../docs/archive/30_rl_credit_report_20261022.md)。本目录包含历史测量实现，不意味着每个设计已通过或应继续运行。

| 文件 | 职责 | 状态 |
|---|---|---|
| `audit_a.py` | 版本指纹、手写信用规则穷举、token片段检查 | 已执行的零GPU审查 |
| `build_review.py` | 选40个焦点状态；构造真实T与已有像素C | 已执行；看图门槛没过，后续配对回执GPU测量未启动 |
| `mine_utilization.py` | 文本规则挖掘，选择另一批40病例 | 候选产生器，不是视觉正确性标签 |
| `build_case_pages.py` | 重放动作、生成学生输入图像审阅页 | 已执行；公开包已补审阅输入PNG，不含题库完整原文 |
| `segment_value.py` | 13例的P0/P1/P1c/P2续跑 | v1历史实现，存在进展报告列出的测量问题；不在此次发布中悄悄修写历史代码 |
| `summarize_segment.py` | 按病例计算均值和bootstrap区间 | 可在CPU上从公开832行逐次结果重算 |
| `segment_value_v2.py` | v2：token对齐切点 P1t/P2t，遮盖正确选项文字的对照 P1m | 已执行（进展报告第 3 节问题 4）；816行 |
| `summarize_segment_v2.py` | v2 汇总（S2m = V(P1t) − V(P1m)） | CPU 可从 `logs/mve/segval_v2/runs_*.jsonl` 重算，结果与 `summary.json` 一致 |
| `export_review_release.py` | 从本地源轨迹及人工写入的模型审阅标签生成公开包 | CPU；白名单导出，按审阅哈希复制PNG，不自动看图或产生标签 |
| `verify_review_release.py` | 验证记录、锚点、PNG哈希、图片链接、逐次结果及汇总 | CPU；结构/数字复核，不是视觉真值验证 |

## 无需GPU的公开复核

在仓库根目录：

```bash
python3 pipeline/eval/mve/verify_review_release.py
# 可选：在临时目录重新生成完整 summary（不覆盖历史结果）
work=$(mktemp -d)
cp logs/mve/segval_v1/runs_*.jsonl "$work/"
python3 pipeline/eval/mve/summarize_segment.py "$work"
```

验证程序只依赖Python标准库。结果快照为 `logs/mve/segval_summary.json`。832行只含已记录的结果字段，原代码没有保存每次续跑的生成文本和完整prompt token IDs。

## 从合法持有的材料重建审阅包

已有源轨迹布局及两批审阅标注时：

```bash
python3 pipeline/eval/mve/export_review_release.py --workspace /path/to/icml
```

工作区需有 `research/trajectories_20261022/{sft_twobranch_300,sft_twobranch_1700,e1_step160,utilization_mining}` 及 `research/annotations/{mve_prep_20261022_codex,utilization_20261022_codex}`。图片输入默认在 `research/review_images_20261022/{receipt,utilization}`，子目录保持审阅页原有的 sXX/uXX 布局；也可用 `--image-root` 指定。PNG必须逐字节匹配原审阅哈希，导出不会压缩或改图。源文件哈希见发布包 `source_manifest.json`。导出将覆盖本仓库 `data/review_20261022` 的派生产物，不改源轨迹或标注。

已发布的图片可直接在逐题Markdown中查看。重新从源数据生成审阅页另需合法源数据、原裁图环境和处理器。按原脚本设置 `CAPTION_GRPO_CODE`、`EASYR1_DIR`、`DATA_DIR`；源目录和检查点不能由公共包自动下载。所需依赖在原环境中包括 pandas、Pillow、transformers、EasyR1，GPU续跑另需vLLM。`build_review.py` 的SRC和 `review_sample.json` 的file字段是原实验的相对布局；外部重建需挂载对应目录，不能只拿题号而用另一张图代替。

公开 `segment_boundaries.json` 去除了question字段；v1测量代码从源数据读取题面，不依赖该字段。它保留原字符锚点，并明确token边界未验证。不要把此版本直接当成修好后的测量运行；新版本应有新的计划、输出目录和版本记录。
