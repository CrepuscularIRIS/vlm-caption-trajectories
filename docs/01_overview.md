# 实验报告（2026-10-24，实验进行中）

本页只写配置、正在进行的实验和统计结果，不写结论和原因解释。还没弄清的地方写成问题，放在 §6。代码层面的已知问题与修复状态见 [docs/02](02_known_issues.md)。之前的全部报告已归档到 [docs/archive/](archive/README.md)。

## 1. 任务

导师的目标（10-18 原话）：
- “希望一个 Base model 最少要具备这些最基础的行为，而且是自主的”；
- “在这个基础上，模型可以达到最佳的 tradeoff between performance and efficiency”。

做法：
- **SFT**：在 Qwen3-VL-4B 上，用我们构造的“短证据 caption + 真实裁图”轨迹训练。
- **RL**：在 SFT 之后用 GRPO 系列方法训练。

## 2. 设置

### 2.1 交互协议（caption v5.1）

每一步的格式：

```text
<think>
Step N | re_examine
[observation_k] 已看到的候选还缺一个区分条件，下一个视图检查它。
</think>
<grounding>{"bbox_2d":[x1,y1,x2,y2],"source":"observation_k"}</grounding>
```

- 动作只有两种：`<grounding>` 裁图，或 `<answer>X</answer>` 终答。
- 裁图框是相对于某个已有视图的 [0,1] 坐标。环境回到原图全分辨率重新裁取，并把新视图返回给模型。
- 最多裁图 8 次。
- 越界或倒置的框会被拒绝，这条轨迹以非法动作结束。
- 定义见 [`code/reference/caption_v51.py`](../code/reference/caption_v51.py)。

### 2.2 SFT

- 库 A 共 4,040 条轨迹（V2.2，多个教师模型生成，经过逐图审核），可在[网页目录](https://CrepuscularIRIS.github.io/vlm-caption-trajectories/)查看。
- SFT 使用其中 3,983 题，共 6,953 个监督轮，训练 2 个 epoch，得到 **A-SFT**。
- 答案式 SFT（H1 的起点）：同样的题和图，监督目标只有最终答案。

### 2.3 RL（标准 GRPO）

| 项 | 值 |
|---|---|
| 题池 | 8,000 题：MME-RealWorld 5,975，MMMU-Pro 1,064，WorldBench 961 |
| 每步 | 128 题 × 16 条 rollout |
| 每步更新 | 4 次（mini-batch 32 题） |
| 学习率 | 1e-6，warmup 5%，之后恒定 |
| clip | .2 / .3 |
| max_grad_norm | 1.0 |
| KL / 动态采样 | 不开 / 不开 |
| 温度 | .7 |
| 奖励 | 答对 1，答错 0 |
| caption 训练的序列预算 | 每轮 640 token，总响应 7,168 token（含工具回执） |
| 无工具训练的序列预算 | 响应 4,096 token |
| 图像上限 | 589,824 像素 |
| seed | 42（单 seed） |
| 硬件 | 每组 2 张 RTX PRO 6000 96GB |

**训练长度：**
- 每 25 步在 440 道验证题上验证一次，并保存完整 checkpoint。
- 从第 150 步起，如果最近 3 次验证的均值比再之前 3 次的均值高不到 0.01，就在保存完 checkpoint 后停止。上限 300 步。
- 这条规则在开跑前写定，代码见 `plateau_stop.py`。

每步时间和配置见 §2.4。

### 2.4 训练时间与配置（还没确定合理的配置）

**每步的实测量**（B4 冒烟，2 张卡）：

| 第几步 | rollout | 旧策略 log-prob | 更新（4 次） | 奖励 | 合计（不含保存） |
|---|---:|---:|---:|---:|---:|
| 1 | 509 s | 181 s | 616 s | 14 s | 1,320 s |
| 2 | 422 s | 178 s | 611 s | 14 s | 1,225 s |

第 1 步共 2,048 条轨迹（128 题 × 16 条）：
- 处理 token 合计 8.79M，生成 token 合计 0.29M；
- 每条轨迹平均裁图 0.72 次。

旧配置（每步 12 题 × 16 条 = 192 条轨迹，2 张卡）每步约 140 秒。

**时间：**

| | B4（A-SFT + GRPO，带裁图） | B6（原模型 + GRPO，无工具） |
|---|---:|---:|
| 每步 | 20–22 分钟 | 约 12 分钟 |
| 到第 150 步（停止规则最早生效） | 约 51–55 小时 | 约 30 小时 |
| 到第 300 步（上限） | 约 102–110 小时 | 约 60 小时 |

另外每 25 步还有一次验证（440 题）和一次 checkpoint 保存。

**部署**：为了能同时做多组实验，每组用 2 张卡，4 张卡同时跑两组（现在是 B6 + 之后的 B4）。

**决定训练开销的设置**（[`grpo_v51fix.sh`](../pipeline/grpo/scripts/grpo_v51fix.sh)）：

```bash
data.train_files=.../grpo_caption_v51_8k/train.parquet   # 8,000 题的题池
data.rollout_batch_size=128                               # 每步 128 题
worker.rollout.n=16                                       # 每题 16 条 rollout
worker.actor.global_batch_size=32                         # mini-batch 32 题 → 每步 4 次更新
data.max_response_length=7168                             # 多轮总响应预算（含工具回执）
worker.actor.micro_batch_size_per_device_for_update=2
worker.actor.micro_batch_size_per_device_for_experience=4
worker.rollout.gpu_memory_utilization=0.45                # 2 卡上 vLLM 与训练共用显存
trainer.n_gpus_per_node=2
trainer.max_steps=300 trainer.val_freq=25 trainer.save_freq=25   # 上限 300 步，每 25 步验证并保存
```

**还不清楚的问题：**
1. 要训练到收敛，每步应该用多少题、每题多少条 rollout、每步更新几次，才算合理？
2. 需要的步数和时间，主要由什么决定：题目的选择（8k 题池的组成；A-SFT 在第 1 步训练题上的 rollout 准确率是 .549），还是 rollout 的配置（128 × 16）？
3. 2 卡一组、两组并行，与 4 卡跑一组相比，哪种更合适？

### 2.5 评测

- **E1**：552 题（440 道分层验证题 + 112 道 V* 留出题），每题采样 4 次，T = .7。
- **预算**：上下文 32,768，累计生成 8,192 token，每轮可以用完剩余预算。
- **评分**：所有条件用同一个脚本。答案取最后一个 `<answer>X</answer>`（`\boxed{}` 和 “answer is X” 作为备选）。截断或出错计为错误。
- **成本**：
  - 首裁率：至少裁过一次的轨迹比例；
  - 后续裁图：每题 max(裁图数 − 1, 0)；
  - 每题处理 token：每轮完整输入（包括重复的历史和图像 token）加生成，累计。
- **数据身份**：
  - 8k 题池含 700 道 MMMU-Pro 官方 test 来源题；
  - A-SFT 的训练数据含官方 V*（191 题）中的 67 题；
  - E1 的 112 道 V* 与这 67 题不重合。

## 3. 代码

| 路径 | 内容 |
|---|---|
| [`pipeline/grpo/v51fix/`](../pipeline/grpo/v51fix/) | RL：多轮裁图 rollout、裁图环境、奖励、逐轨迹成本记录 |
| [`pipeline/grpo/scripts/grpo_v51fix.sh`](../pipeline/grpo/scripts/grpo_v51fix.sh) | caption GRPO 启动脚本（§2.3 的参数） |
| [`pipeline/grpo/scripts/grpo_v51fix_notool.sh`](../pipeline/grpo/scripts/grpo_v51fix_notool.sh) | 无工具 GRPO 启动脚本 |
| [`pipeline/grpo/easyr1_602a128_local.diff`](../pipeline/grpo/easyr1_602a128_local.diff) | 对 EasyR1 的本地改动 |
| [`pipeline/eval/baselines/eval_base.py`](../pipeline/eval/baselines/eval_base.py) | 原模型评测：不带工具 / 原生 zoom 工具 |
| [`pipeline/eval/baselines/eval_caption.py`](../pipeline/eval/baselines/eval_caption.py) | caption 协议评测 |
| [`pipeline/eval/baselines/summarize.py`](../pipeline/eval/baselines/summarize.py) | 统一评分与成本统计 |

### 关键代码（节选）

奖励：只看唯一一个 `<answer>` 是否等于标准答案（[`reward_caption.py`](../pipeline/grpo/v51fix/reward_caption.py)）：

```python
def _norm_answer(text):
    matches = _ANS.findall(text)            # <answer>...</answer>
    if len(matches) != 1:                   # 轨迹在第一个答案处结束；多个答案算无效
        return None
    answer = matches[0].strip().upper()
    return answer if _LETTER.fullmatch(answer) else None   # A–J

def score_one(data_source, solution_str, ground_truth, extra_info=None):
    gold = str(ground_truth).strip().upper()
    return float(bool(_LETTER.fullmatch(gold)) and _norm_answer(solution_str) == gold)
```

裁图框检查：不合法的框直接拒绝，不做截断或夹边（[`caption_env.py`](../pipeline/grpo/v51fix/caption_env.py)）：

```python
def valid_bbox(b):
    x0, y0, x1, y1 = (float(v) for v in b)
    return 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
```

停止规则（`plateau_stop.py`，开跑前写定）：

```python
MIN_STEP, WINDOW, MIN_GAIN = 150, 3, 0.01
# vals = [(step, val_accuracy), ...]，每 25 步一个点
if vals[-1][0] >= MIN_STEP and len(vals) >= 2 * WINDOW:
    gain = mean(vals[-3:]) - mean(vals[-6:-3])
    if gain < MIN_GAIN:
        ...  # 等这一步的 checkpoint 保存完，再停止本 run 的进程
```

## 4. 正在进行的实验

| ID | 条件 | GPU | 状态 |
|---|---|---|---|
| B0 | 原模型，不带工具的 prompt | — | **完成** |
| B1 | 原模型 + 原生 zoom 工具 | 1 | 评测中（13:00 时 384/552 题） |
| B2 | 原模型 zero-shot + caption 协议 | — | **完成** |
| B3 | A-SFT | — | **完成** |
| H1 | 答案式 SFT | 0 | 评测中（13:00 开始） |
| H2 | 答案式 SFT + GRPO（旧训练第 160 步 checkpoint） | 0 | H1 之后评测 |
| B6 | 原模型 + GRPO，无工具 | 2, 3 | **训练中**（10-24 12:50 开始，见 §5.2） |
| B4 | A-SFT + GRPO | 0, 1 | 排队：B1 和 H2 评测结束后自动开始训练（见 §5.2） |
| B5 | 原模型直接 GRPO（caption 协议） | — | B6 之后 |
| B7 / B8 | AdaptVision / CounterCredit（同底座适配） | — | 未开始 |
| X1 / X2 | Mini-o3-7B、DeepEyes-7B | — | 未开始（Mini-o3 权重已下载） |

## 5. 结果

### 5.1 E1（552 题 × 4 次采样 = 2,208 条轨迹）

| ID | 准确率 | 有 `<answer>` 标签 | 首裁率 | 后续裁图 / 题 | 处理 token / 题 | 生成 token / 题 | 截断率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | .5217 | .9534 | 0 | 0 | 1,464 | 810 | .0466 |
| B1 | 评测中 | | | | | | |
| H1 | 评测中 | | | | | | |
| H2 | 评测中 | | | | | | |
| B2 | .0068 | .0149 | .0145 | .0747 | 2,386 | 105 | .0005 |
| B3 | .5670 | .9805 | .5725 | .4216 | 5,015 | 170 | .0014 |

![E1 trade-off](../media/baselines_v52_20261024/tradeoff_E1.png)

图中目前只有已完成的条件；其余条件完成后加入。汇总数据：[`logs/baselines_v52_20261024/summary.json`](../logs/baselines_v52_20261024/summary.json)，作图脚本：[`plot_tradeoff.py`](../pipeline/eval/baselines/plot_tradeoff.py)。

按来源统计的准确率：

| ID | MME-RealWorld | MMMU-Pro | V* | WorldBench |
|---|---:|---:|---:|---:|
| B0 | .5273 | .3397 | .6741 | .3306 |
| B2 | .0044 | .0449 | .0045 | .0000 |
| B3 | .5708 | .2885 | .8125 | .2782 |

B2 的 2,208 条轨迹按最后一轮分类：

| 最后一轮 | 条数 |
|---|---:|
| 裁图请求，但 bbox 不在 [0,1] 内，被拒绝（例如 `[355,122,595,152]`） | 1,021 |
| 既没有裁图动作也没有 `<answer>` 标签 | 830 |
| `<answer>` 终答 | 357 |

### 5.2 训练（进行中）

| 组 | GPU | 开始时间 | 每步时间（冒烟第 1 步） | 第 150 步（最早可停） | 第 300 步（上限） |
|---|---|---|---:|---:|---:|
| B6 原模型 + GRPO，无工具 | 2, 3 | 10-24 12:50 | 约 12 分钟 | 约 30 小时 | 约 60 小时 |
| B4 A-SFT + GRPO | 0, 1 | 排队（B1 与 H2 评测结束后） | 20–22 分钟 | 约 51–55 小时 | 约 102–110 小时 |

- 时间均为北京时间。
- 天数按冒烟第 1 步的耗时乘以步数折算，**另外**每 25 步还有一次验证和一次 checkpoint 保存。实际时间以训练日志为准。
- 每 25 步的验证准确率出来后填在这里。

## 6. 还没弄清的问题

训练配置与时间的问题见 §2.4。

1. 原模型在 caption 协议下，为什么近一半轨迹给出 0–1000 尺度的框，另有 830 条既不裁图也不写 `<answer>`？
2. B4 和 B6 在多少步达到平台？平台时的准确率和裁图成本是多少？
3. 不做 SFT 的 B5，在 caption 协议下能否得到非零奖励？
