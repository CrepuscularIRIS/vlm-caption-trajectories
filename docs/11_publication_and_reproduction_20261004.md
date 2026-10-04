# 本次发布范围、来源与复算

本次是公开研究进展与导师审阅包，**不是 32,333 条轨迹的数据集发布，也不是 1,241 条可直接训练的 SFT 文件**。

后续同日增补：[cc 补审完成与 75 条抽检](12_cc_spotcheck_update_20261004.md)。最终自动候选变为 1,317 条，仍未作为已验收训练集分发。新增 75 条的完整可见文本、历史 69 条版本、273 份脱敏回执索引与拒绝分流清单；没有新增第三方图片。原图、题目原文与完整 CLI stdout 仍仅在私有源工作区。

## 包含什么

- 2026-10-03 23:10（America/New_York）的量产/审核汇总，2,520 条脱敏审核索引、81 条漏审 fallback ID；结果与审核源文件哈希保留。
- 9 题、18 条真实完整可见轨迹，包含正例候选、过程反例、恢复案例与金标争议。不是随机抽样；人工终裁均未填写。
- 8 个来自 4 道 WorldBench 题的去重输入视图，公开页面直接可看。
- 五个量产 Python 文件的原样阅读快照，加 TeacherContext/GUIDANCE 的明确摘录；标准库复算和本地带图案例册脚本。
- 导师要求的公开转述、当前缺口和五个具体问题。没有发布私人讨论原话。

没有发布密钥、私有服务地址、完整 API 请求/响应或隐藏推理、完整原始数据集、模型权重、运行账户与调度配置。案例 JSON 中 `visible_output` 是保存的显式 caption/动作，可能已由原生成器做格式标签归一；不改写其观察内容。

## 图像来源和归属

|来源|公开图像范围|依据|
|---|---|---|
|WorldBench|仅 `worldbench_0826`、`1913`、`0675`、`0841` 的 8 个保存视图|官方 [数据卡](https://huggingface.co/datasets/zlab-princeton/WorldBench) 标注 [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)|
|MME-RealWorld|不发布原图或 crop；任务用中文转述，保留模型可见证据文字、动作、来源 ID 与哈希|上游 [Dataset License](https://github.com/MME-Benchmarks/MME-RealWorld#dataset-license) 限学术研究，并限制未经许可的分发/发布|

WorldBench 归属：**WorldBench dataset contributors / Zlab Princeton**，作者与数据详情以链接的数据卡为准。本仓库并非该数据集作者。保存输入视图在原始运行中经过缩放或裁剪；本次从已保存文件逐字节复制，未再修改图片、添加标注或合成内容。每张图片的原数据图像哈希、保存视图哈希、题目 ID 与修改说明在 [ATTRIBUTION.json](../media/20261004/ATTRIBUTION.json)。不将数据许可扩展为原代码、全部图源或整个仓库的统一许可。

公开图片足以查看 C01、C07–C09；C02–C06 必须在合法持有原数据的本地工作区打开完整图版，不能声称公开包已让读者独立完成全部像素核验。任何公共展示中的原始金标均按源记录保留，尤其 C08/C09 没有基于模型多数票改标。

## 无网络复算

Python 3.10+，只用标准库，不调用模型或 GPU：

```bash
python scripts/verify_snapshot.py
python scripts/inspect_case.py --all
python scripts/verify_production.py
python scripts/build_casebook.py --check-markdown
python scripts/spotcheck_review.py --verify
```

前两个验证历史 30 题/16 条种子快照与旧案例；后两个验证新批次池统计、图组/步数/材料分布、候选标记、案例时序、图片字节和案例册文字对应。`manifest.sha256.json` 覆盖所有发布文件（不自包含）。**这些检查不判断图片事实、人工验收、金标真伪或训练增益。** 量产总量来自冻结统计；公开包没有分发全部原轨迹，因此不能仅靠本仓库独立重算每次生成的完整过程。

## 本地完整图版

在拥有对应源文件的机器上，把 `--workspace-root` 指向包含 `obs/runs/` 的研究工作区。输出必须在公开 Git 仓库之外：

```bash
python scripts/build_casebook.py \
  --workspace-root /path/to/research-workspace \
  --output /path/to/private-review/CASEBOOK.html
```

脚本按相对路径定位输入图，逐个校验 SHA-256，复制到输出旁的 `views/` 后生成静态 HTML。图片按实际动作时序出现，caption 前明确列出已收到的视图。文件缺失或哈希不匹配会失败，不会联网补图或偷偷替换输入。此输出含 MME 原图，**不能整体上传公开仓库**。

本次本地构建已校验 22 个去重输入视图；“22 张 hash 一致”不等于 22 张事实全部正确。人工意见仍留待实际填写。

新 75 条抽检可按哈希复制既有完整看图册到单独的私有审阅目录，不运行模型或重抽样：

```bash
python scripts/spotcheck_review.py \
  --workspace-root /path/to/research-workspace \
  --output /path/to/private-review/SPOTCHECK_CASEBOOK.html
```

该 HTML 含第三方图片，只供本地研究审阅，不应上传公开仓库。公开的 [75 条文本册](SPOTCHECK75_20261004.md) 可直接用于索引；人工填写原生产目录的终裁表，不能把公开文本元数据的 PENDING 改成虚构 KEEP。

## 可追溯性与版本边界

[production_provenance_20261004.json](../data/production_provenance_20261004.json) 保存私有源文件相对路径/hash、代码原样/摘录方式和历史核对摘要；不意味着这些私有文件全部公开。2,520 行索引只保留分池、行为特征和哈希，不含原始题图或 API 全量数据。

本次公开改动未改变生成器、审核器、金标、原始轨迹、人工表或运行队列；已知缺口如实保留。后续修复、补审、人工接受和训练实验应分别记录，不能用文档更新代替执行完成。
