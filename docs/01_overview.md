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
| 题池 | 6,016 题：MME-RealWorld 4,577，WorldBench 790，MMMU-Pro 649（与 SFT 不重合，见下） |
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
- 步数 = 题池大小 ÷ 每步题数 = 6,016 ÷ 128 = **47 步**。
- 题池取 6,016 = 47 × 128：EasyR1 会丢掉最后不满一批的题（`drop_last=True`），取整后这一遍不丢题。
- 第 25 步和最后一步各验证一次（440 道验证题），并保存完整 checkpoint。
- 启动参数：`trainer.total_epochs=1`（不设 `max_steps`），warmup 是 47 步的 5%。

**题池构建**（[`pool6k.py`](../pipeline/grpo/scripts/pool6k.py)，重叠检查 [`sft_overlap.py`](../pipeline/grpo/scripts/sft_overlap.py)）：
- 候选：v51 训练池去掉 V*、E1（uid / 图片路径 / 16×16 均值哈希）和 v51 验证集，余 13,074 题。
- 去掉与 A-SFT 数据（3,983 题 / 4,040 条轨迹）任一项相同的题：题面、题面加选项、原图 16×16 均值哈希（同一张图问不同问题也去掉）。去掉 1,610 题。
- 旧 2k 题池中剩下的 1,867 题全部保留，其余按 seed 42 的顺序补足到 6,016 题，同一图像哈希只取一题。
- 与 A-SFT 的重叠（同一脚本统计）：

| 数据 | 题数 | 题面相同 | 题面加选项相同 | 原图哈希相同 |
|---|---:|---:|---:|---:|
| 6k 题池（本次训练） | 6,016 | 0 | 0 | 0 |
| 8k 题池（10-24 13:53 的两组，已停） | 8,000 | 0 | 0 | 961 |
| 验证集（训练中每次验证用） | 440 | 0 | 0 | 18 |

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

**时间**（实测每步时间，来自 10-24 的 8k 两组；一遍 = 47 步，另有验证和保存）：

| | B4（A-SFT + GRPO，带裁图） | B6（原模型 + GRPO，无工具） |
|---|---:|---:|
| 每步（实测中位数） | 20.6 分钟（1,239 s） | 11.8 分钟（707 s） |
| 一遍（47 步） | 约 16.7 小时 | 约 9.7 小时 |

**部署**：每组 2 张卡，两组同时跑。目前每组的 2 张卡都在 rollout（vLLM）和训练（FSDP）之间轮流使用。

**决定训练开销的设置**（[`grpo_v51fix.sh`](../pipeline/grpo/scripts/grpo_v51fix.sh)，加上一遍的覆盖参数）：

```bash
data.train_files=.../grpo_caption_v51_6k/train.parquet   # 6,016 题的题池
data.rollout_batch_size=128                               # 每步 128 题
worker.rollout.n=16                                       # 每题 16 条 rollout
worker.actor.global_batch_size=32                         # mini-batch 32 题 → 每步 4 次更新
data.max_response_length=7168                             # 多轮总响应预算（含工具回执）
worker.actor.micro_batch_size_per_device_for_update=2
worker.actor.micro_batch_size_per_device_for_experience=4
worker.rollout.gpu_memory_utilization=0.45                # vLLM 与训练共用显存
trainer.n_gpus_per_node=2
trainer.total_epochs=1 trainer.max_steps=null             # 题池过一遍：6016 // 128 = 47 步
trainer.val_freq=25 trainer.save_freq=25
```

**导师 10-24 的回复（此后作为通用 RL 配置）：**

> 训练时间主要是和我们在RL阶段投入的样本数对齐，所以取决于这个设定。假设我们现在rl 2k samples，按现在的设定其实150+ step就已经把每道题都过了一遍。不同于sft阶段，假设sft训练2epoch就是把相同的训练样本重复过两遍，rl阶段通常不需要重复去过题库。那么这种情况下只要你的rl steps能够覆盖所有rl samples，就算跑完了一个周期。你可以把rl samples样本量提上去，然后可以算一下所需要的steps，只需要跑满就可以了，不用重复刷。另外2 * 2的配置是合理的，一组双卡然后里面一张做rollout一张做反向传播，这个是没问题的。

题池大小：先选 8k（10-24 13:53 启动），同日改为 6k（17:58 启动），并要求与 SFT 数据完全不重合。

### 2.5 评测

- **E1**：552 题（440 道分层验证题 + 112 道 V* 留出题），每题采样 4 次，T = .7。
- **预算**：上下文 32,768，累计生成 8,192 token，每轮可以用完剩余预算。
- **评分**：所有条件用同一个脚本。答案取最后一个 `<answer>X</answer>`（`\boxed{}` 和 “answer is X” 作为备选）。截断或出错计为错误。
- **成本**：
  - 首裁率：至少裁过一次的轨迹比例；
  - 后续裁图：每题 max(裁图数 − 1, 0)；
  - 每题处理 token：每轮完整输入（包括重复的历史和图像 token）加生成，累计。
- **数据身份**：
  - 6k 题池含 433 道 MMMU-Pro 官方 test 来源题（8k 题池为 700 道）；
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
| [`pipeline/eval/baselines/eval_minio3.py`](../pipeline/eval/baselines/eval_minio3.py) | X1：Mini-o3 官方协议（`<grounding>` 相对坐标，裁原图不放大），E1 预算 |
| [`pipeline/eval/baselines/eval_deepeyes.py`](../pipeline/eval/baselines/eval_deepeyes.py) | X2：DeepEyes 官方协议（`image_zoom_in_tool` 绝对像素坐标），E1 预算 |
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
self.training_steps = len(train_dataloader) * config.trainer.total_epochs   # 6016 // 128 * 1 = 47
```

## 4. 正在进行的实验

| ID | 条件 | GPU | 状态 |
|---|---|---|---|
| B0 | 原模型，不带工具的 prompt | — | **完成** |
| B1 | 原模型 + 原生 zoom 工具 | — | **完成** |
| B2 | 原模型 zero-shot + caption 协议 | — | **完成** |
| B3 | A-SFT | — | **完成** |
| H1 | 答案式 SFT | — | **完成** |
| H2 | 答案式 SFT + GRPO（旧训练第 160 步 checkpoint） | — | **完成**（旧规模：每步 12 题，160 步） |
| H2（6k） | 答案式 SFT + 答案式协议 GRPO，6k 题池一遍 | 2, 3 | 排队（B5 之后） |
| B6 | 原模型 + GRPO，无工具 | 2, 3 | **训练完成**（10-24 17:58 – 10-25 01:52，6k 题池一遍 47 步，见 §5.2）；第 47 步 E1 评测**完成**（§5.1） |
| B4 | A-SFT + GRPO | 0, 1 | **训练中**（10-25 02:08 重启，`b4_6k_m1`：update micro batch 2 → 1，其余不变；验证集 422 题）。第一次（`b4_6k`）在第 7 步 CUDA OOM 退出，见 §5.2 |
| B5 | 原模型直接 GRPO（caption 协议），6k 题池一遍 | 2, 3 | **训练中**（10-25 04:45 开始，47 步，验证集 422 题） |
| B7 / B8 | AdaptVision / CounterCredit（同底座适配） | — | 未开始 |
| X1 | Mini-o3-7B-v1（官方协议，E1 预算） | — | **完成**（§5.1） |
| X2 | DeepEyes-7B（官方协议，E1 预算） | — | **完成**（§5.1） |

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
| B6 | .5960 | .9932 | 0 | 0 | 1,113 | 459 | .0068 |
| X1 | .6395 | .9398 | .8927 | 1.8107 | 17,280 | 680 | .0059 |
| X2 | .5589 | .9035 | .8483 | .0041 | 13,800 | 3,292 | .1051 |

H1、H2 使用答案式协议评测（工具可用，2,208 条轨迹中裁图次数均为 0）。

![E1 trade-off](../media/baselines_v52_20261024/tradeoff_E1.png)

图中是 B0–B3、H1、H2、B6、X1、X2 九个条件，其余条件完成后加入。B6 为 6k 题池一遍（47 步）的最后一个 checkpoint，使用 B0 的不带工具 prompt 评测。X1 为 Mini-o3-7B-v1（Qwen2.5-VL-7B 底座），按其官方协议评测，图像输入上限 2,000,000 像素（我们的条件为 589,824 像素），每题最多 12 张图；2,208 条轨迹中 115 条因达到 12 张图上限结束，0 条工具调用报错。X2 为 DeepEyes-7B（Qwen2.5-VL-7B 底座），按其官方 `eval_vstar.py` 协议评测（原图不缩放输入，裁图用绝对像素坐标）；2,208 条轨迹中 230 条截断、2 条工具调用出错、2 条超出上下文。官方代码没有 `</tool_call>` 停止词，我们也没有加；截断的轨迹中可见 `</tool_call>` 之后继续生成、重复工具调用 JSON 直到用完 8,192 生成预算的情况。汇总数据：[`logs/baselines_v52_20261024/summary.json`](../logs/baselines_v52_20261024/summary.json)，作图脚本：[`plot_tradeoff.py`](../pipeline/eval/baselines/plot_tradeoff.py)。

按来源统计的准确率：

| ID | MME-RealWorld | MMMU-Pro | V* | WorldBench |
|---|---:|---:|---:|---:|
| B0 | .5273 | .3397 | .6741 | .3306 |
| B1 | .6084 | .3077 | .7902 | .2944 |
| B2 | .0044 | .0449 | .0045 | .0000 |
| B3 | .5708 | .2885 | .8125 | .2782 |
| H1 | .4683 | .2885 | .5156 | .3468 |
| H2 | .5125 | .2564 | .5692 | .2742 |
| B6 | .6364 | .4103 | .6920 | .3185 |
| X1 | .6364 | .3462 | .9085 | .3548 |
| X2 | .5678 | .2500 | .7746 | .3145 |

B2 的 2,208 条轨迹按最后一轮分类：

| 最后一轮 | 条数 |
|---|---:|
| 裁图请求，但 bbox 不在 [0,1] 内，被拒绝（例如 `[355,122,595,152]`） | 1,021 |
| 既没有裁图动作也没有 `<answer>` 标签 | 830 |
| `<answer>` 终答 | 357 |

### 5.2 训练

| 组 | GPU | 开始时间 | 步数 | 每步时间 | 预计完成 |
|---|---|---|---:|---:|---|
| B6 原模型 + GRPO，无工具 | 2, 3 | 10-24 17:58 | 47 | 588 秒（均值） | 10-25 01:52 完成 |
| B4 A-SFT + GRPO | 0, 1 | 10-24 17:58 | 完成 6 步后退出 | 约 21 分钟 | — |

验证准确率（440 道验证题，`accuracy_reward`）：

| 组 | 第 0 步 | 第 25 步 | 第 47 步 |
|---|---:|---:|---:|
| B6 | .470 | .557 | .564 |
| B4（`b4_6k`，440 题） | .502 | — | — |
| B4（`b4_6k_m1`，422 题） | .533 | | |

- B4 在 20:39 第 7 步的 update_actor 中退出：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.41 GiB. GPU 0 ... 6.21 GiB is free ... this process has 88.73 GiB memory in use`。没有 checkpoint（第一次保存在第 25 步）。是否重启、配置是否改动待定。

- 时间均为北京时间。预计完成时间 = 每步时间 × 47，再加上验证和保存；实际以训练日志为准。
- 之前按“到收敛为止”启动的两组（12:50 / 13:18）已在导师 10-24 回复后停止，没有使用它们的结果。
- 8k 题池的两组（13:53 启动）在 17:57 停止（B4 完成 10 步，B6 完成 21 步），没有使用它们的结果。
- B6 checkpoint：`global_step_25`、`global_step_47`（EasyR1 记录的最佳验证步为 47，.5636）。

## 6. 还没弄清的问题


1. 原模型在 caption 协议下，为什么近一半轨迹给出 0–1000 尺度的框，另有 830 条既不裁图也不写 `<answer>`？
2. B4 和 B6 过完一遍题池后，准确率和裁图成本分别是多少？
3. 不做 SFT 的 B5，在 caption 协议下能否得到非零奖励？
