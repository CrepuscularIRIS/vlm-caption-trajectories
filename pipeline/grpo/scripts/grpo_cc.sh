#!/bin/bash
# 卡 A / 卡 B（2026-10-18）：CounterCredit 价格通道（arXiv 2609.22910）适配到 caption v5.1。
# 奖励 = 正确与否；每次裁图在调用前状态上做 立刻作答 / 真实裁图 / 3 个随机同尺寸块 探针，
# 价格 P = min(C, γΣ(q−ε)) − ρ·未验证裁图数，优势 = GRPO(正确) + (P − 组均值)，价格不除 std。
# 从 A 的 SFT checkpoint 起跑，165 步；其余超参与 grpo_cropfloor.sh / grpo_2plus2.sh 的 A 分支一致。
# 用法: bash grpo_cc.sh <TAG> <GPUS> [额外 override...]
#   卡 A: nohup bash grpo_cc.sh cc 0,1 > outputs/grpo_v3_A_cc.log 2>&1 &
#   卡 B: nohup bash grpo_cc.sh cc_dsasym 2,3 algorithm.online_filtering=true algorithm.filter_key=accuracy \
#           algorithm.filter_low=0.01 algorithm.filter_high=1.01 > outputs/grpo_v3_A_cc_dsasym.log 2>&1 &
# 正式运行时另设 CC_EPS=0.2（SFT-A 冷启动上的零假设校准，见 v51_cc/calibrate_eps.py）。
set -e
TAG=$1; GPUS=$2
shift 2 2>/dev/null || true
[ -n "$TAG" ] && [ -n "$GPUS" ] || { echo "usage: $0 TAG GPUS [overrides]"; exit 2; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=${CACHE_DIR}/hf
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd ${EASYR1_DIR}
export RAY_TMPDIR=/tmp/ray_cc_$TAG
export CAPTION_ENGINE_KWARGS="enable_prefix_caching=true,disable_log_stats=false"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAPTION_GRPO_CODE=${PROJECT_DIR}/grpo/v51_cc
export CUDA_VISIBLE_DEVICES=$GPUS
export CAPTION_ROLLOUT_LOG=${OUTPUT_DIR}/grpo_v3_A_${TAG}_rollouts.jsonl
POOL=${DATA_DIR}/grpo_caption_v51_v3
MODEL=${OUTPUT_DIR}/sft_v22_turnwise_merged
NAME=grpo_v3_A_$TAG
CKPT=${OUTPUT_DIR}/grpo_v3_A_$TAG
export PATH=${EASYR1_ENV}/bin:$PATH
python $CAPTION_GRPO_CODE/easyr1_caption.py \
    config=examples/config.yaml \
    data.train_files=$POOL/train.parquet \
    data.val_files=$POOL/val.parquet \
    data.prompt_key=prompt data.image_key=images data.answer_key=reward_model \
    data.max_prompt_length=4096 data.max_response_length=7168 \
    data.max_pixels=589824 data.min_pixels=1024 \
    data.rollout_batch_size=12 data.val_batch_size=192 data.seed=42 \
    data.filter_overlong_prompts=false \
    algorithm.adv_estimator=grpo algorithm.kl_coef=0.0 algorithm.disable_kl=true \
    algorithm.online_filtering=false \
    worker.actor.global_batch_size=12 worker.actor.max_grad_norm=0.1 \
    worker.actor.model.model_path=$MODEL worker.actor.model.trust_remote_code=true \
    worker.actor.optim.lr=1.0e-6 worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.optim.lr_scheduler_type=constant \
    worker.actor.on_policy_reuse_log_probs=true \
    worker.actor.offload.offload_params=false worker.actor.offload.offload_optimizer=false \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.n=16 worker.rollout.temperature=0.7 worker.rollout.top_p=1.0 \
    worker.rollout.limit_images=10 worker.rollout.gpu_memory_utilization=0.45 \
    worker.rollout.tensor_parallel_size=1 worker.rollout.max_num_batched_tokens=32768 \
    worker.rollout.disable_tqdm=true \
    worker.reward.reward_function=$CAPTION_GRPO_CODE/reward_caption.py:compute_score \
    trainer.total_epochs=1 trainer.max_steps=165 \
    trainer.project_name=caption_2p2 trainer.experiment_name=$NAME \
    trainer.logger='["console"]' trainer.n_gpus_per_node=2 trainer.nnodes=1 \
    trainer.val_freq=40 trainer.val_generations_to_log=0 \
    trainer.save_freq=40 \
    trainer.save_checkpoint_path=$CKPT \
    "${@:1}"
