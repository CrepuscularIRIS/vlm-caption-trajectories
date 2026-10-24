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

**训练长度：RL 题池每道题过一遍，不重复。**
- 步数 = 题池大小 ÷ 每步题数 = 8,000 ÷ 128 → **62 步**。
- EasyR1 会丢掉最后不满一批的题，所以一遍覆盖 7,936 题，有 64 题（0.8%）不在这一遍里。
- 第 25、50 步和最后一步各验证一次（440 道验证题），并保存完整 checkpoint。
- 启动参数：`trainer.total_epochs=1`（不设 `max_steps`），warmup 是 62 步的 5%。

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

**时间**（一遍 = 62 步，另有 3 次验证和 3 次保存）：

| | B4（A-SFT + GRPO，带裁图） | B6（原模型 + GRPO，无工具） |
|---|---:|---:|
| 每步 | 20–22 分钟 | 10–12 分钟 |
| 一遍（62 步） | 约 21–23 小时 | 约 10.5–12.5 小时 |

**部署**：每组 2 张卡，两组同时跑。目前每组的 2 张卡都在 rollout（vLLM）和训练（FSDP）之间轮流使用。

**决定训练开销的设置**（[`grpo_v51fix.sh`](../pipeline/grpo/scripts/grpo_v51fix.sh)，加上一遍的覆盖参数）：

```bash
data.train_files=.../grpo_caption_v51_8k/train.parquet   # 8,000 题的题池
data.rollout_batch_size=128                               # 每步 128 题
worker.rollout.n=16                                       # 每题 16 条 rollout
worker.actor.global_batch_size=32                         # mini-batch 32 题 → 每步 4 次更新
data.max_response_length=7168                             # 多轮总响应预算（含工具回执）
worker.actor.micro_batch_size_per_device_for_update=2
worker.actor.micro_batch_size_per_device_for_experience=4
worker.rollout.gpu_memory_utilization=0.45                # vLLM 与训练共用显存
trainer.n_gpus_per_node=2
trainer.total_epochs=1 trainer.max_steps=null             # 题池过一遍：8000 // 128 = 62 步
trainer.val_freq=25 trainer.save_freq=25
```

**导师 10-24 的回复（此后作为通用 RL 配置）：**

> 训练时间主要是和我们在RL阶段投入的样本数对齐，所以取决于这个设定。假设我们现在rl 2k samples，按现在的设定其实150+ step就已经把每道题都过了一遍。不同于sft阶段，假设sft训练2epoch就是把相同的训练样本重复过两遍，rl阶段通常不需要重复去过题库。那么这种情况下只要你的rl steps能够覆盖所有rl samples，就算跑完了一个周期。你可以把rl samples样本量提上去，然后可以算一下所需要的steps，只需要跑满就可以了，不用重复刷。另外2 * 2的配置是合理的，一组双卡然后里面一张做rollout一张做反向传播，这个是没问题的。

题池大小选 8k（过滤后可用的题共 13,074 道）。

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

训练步数（EasyR1 的计算，`ray_trainer.py`；训练 dataloader 设置 `drop_last=True`）：

```python
# 不设 max_steps 时
self.training_steps = len(train_dataloader) * config.trainer.total_epochs   # 8000 // 128 * 1 = 62
```

## 4. 正在进行的实验

| ID | 条件 | GPU | 状态 |
|---|---|---|---|
| B0 | 原模型，不带工具的 prompt | — | **完成** |
| B1 | 原模型 + 原生 zoom 工具 | — | **完成** |
| B2 | 原模型 zero-shot + caption 协议 | — | **完成** |
| B3 | A-SFT | — | **完成** |
| H1 | 答案式 SFT | — | **完成** |
| H2 | 答案式 SFT + GRPO（旧训练第 160 步 checkpoint） | — | **完成** |
| B6 | 原模型 + GRPO，无工具 | 2, 3 | **训练中**（10-24 13:53 开始，题池一遍 62 步，见 §5.2） |
| B4 | A-SFT + GRPO | 0, 1 | **训练中**（10-24 13:53 开始，题池一遍 62 步，见 §5.2） |
| B5 | 原模型直接 GRPO（caption 协议） | — | B6 之后 |
| B7 / B8 | AdaptVision / CounterCredit（同底座适配） | — | 未开始 |
| X1 / X2 | Mini-o3-7B、DeepEyes-7B | — | 未开始（Mini-o3 权重已下载） |

## 5. 结果

### 5.1 E1（552 题 × 4 次采样 = 2,208 条轨迹）

| ID | 准确率 | 有 `<answer>` 标签 | 首裁率 | 后续裁图 / 题 | 处理 token / 题 | 生成 token / 题 | 截断率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | .5217 | .9534 | 0 | 0 | 1,464 | 810 | .0466 |
| B1 | .5888 | .9348 | .9873 | .7097 | 4,288 | 395 | .0127 |
| B2 | .0068 | .0149 | .0145 | .0747 | 2,386 | 105 | .0005 |
| B3 | .5670 | .9805 | .5725 | .4216 | 5,015 | 170 | .0014 |
| H1 | .4515 | .9995 | 0 | 0 | 1,880 | 11 | 0 |
| H2 | .4792 | 1.0000 | 0 | 0 | 1,880 | 11 | 0 |

H1、H2 使用答案式协议评测（工具可用，2,208 条轨迹中裁图次数均为 0）。

![E1 trade-off](../media/baselines_v52_20261024/tradeoff_E1.png)

图中是 B0–B3、H1、H2 六个条件；训练组（B4、B6）完成后加入。汇总数据：[`logs/baselines_v52_20261024/summary.json`](../logs/baselines_v52_20261024/summary.json)，作图脚本：[`plot_tradeoff.py`](../pipeline/eval/baselines/plot_tradeoff.py)。

按来源统计的准确率：

| ID | MME-RealWorld | MMMU-Pro | V* | WorldBench |
|---|---:|---:|---:|---:|
| B0 | .5273 | .3397 | .6741 | .3306 |
| B1 | .6084 | .3077 | .7902 | .2944 |
| B2 | .0044 | .0449 | .0045 | .0000 |
| B3 | .5708 | .2885 | .8125 | .2782 |
| H1 | .4683 | .2885 | .5156 | .3468 |
| H2 | .5125 | .2564 | .5692 | .2742 |

B2 的 2,208 条轨迹按最后一轮分类：

| 最后一轮 | 条数 |
|---|---:|
| 裁图请求，但 bbox 不在 [0,1] 内，被拒绝（例如 `[355,122,595,152]`） | 1,021 |
| 既没有裁图动作也没有 `<answer>` 标签 | 830 |
| `<answer>` 终答 | 357 |

### 5.2 训练（进行中）

| 组 | GPU | 开始时间 | 步数 | 每步时间 | 预计完成 |
|---|---|---|---:|---:|---|
| B6 原模型 + GRPO，无工具 | 2, 3 | 10-24 13:53 | 62 | 10–12 分钟 | 10-25 约 00:40–03:00 |
| B4 A-SFT + GRPO | 0, 1 | 10-24 13:53 | 62 | 20–22 分钟 | 10-25 约 11:00–13:00 |

- 时间均为北京时间。预计完成时间 = 每步时间 × 62，再加上 3 次验证和保存；实际以训练日志为准。
- 之前按“到收敛为止”启动的两组（12:50 / 13:18）已在导师 10-24 回复后停止，没有使用它们的结果。
- 第 25、50、62 步的验证准确率出来后填在这里。

## 6. 还没弄清的问题


1. 原模型在 caption 协议下，为什么近一半轨迹给出 0–1000 尺度的框，另有 830 条既不裁图也不写 `<answer>`？
2. B4 和 B6 过完一遍题池后，准确率和裁图成本分别是多少？
3. 不做 SFT 的 B5，在 caption 协议下能否得到非零奖励？
