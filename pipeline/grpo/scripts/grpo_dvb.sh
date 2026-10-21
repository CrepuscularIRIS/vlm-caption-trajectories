#!/bin/bash
# Direction B (docs/29 6.4): original GRPO-A recipe (reward = correctness, LAMBDA=0, no KL, no DS) + two-branch value term,
# entering the advantage by CAPTION_DV_MODE (replace_first_centred | add_all_centred | add_first); plugin dir grpo_v51_dvb.
# Usage: CAPTION_DV_MODE=<mode> bash grpo_dvb.sh <TAG> <LAMBDA> <GPUS> [overrides]
# GRPO A + 裁图代价第二轮：reward = correct * (1.0 - λ * crops)，优势分母 max(组内std, 0.25)
# 从 A 的 SFT checkpoint (sft_v22_turnwise_merged) 起跑，165 步；
# 除奖励、优势分母与逐条记录外，超参与 grpo_2plus2.sh 的 A 分支完全一致。
# 用法: bash grpo_cropfloor.sh <TAG> <LAMBDA> <GPUS>，例如 l050 0.05 2,3
# 启动: nohup bash grpo_cropfloor.sh l050 0.05 2,3 > outputs/grpo_v3_A_floor_l050.log 2>&1 &
set -e
TAG=$1; LAMBDA=$2; GPUS=$3
shift 3 2>/dev/null || true
[ -n "$TAG" ] && [ -n "$LAMBDA" ] && [ -n "$GPUS" ] || { echo "usage: $0 TAG LAMBDA GPUS"; exit 2; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=${CACHE_DIR}/hf
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd ${EASYR1_DIR}
export RAY_TMPDIR=/tmp/ray_dvb_$TAG
export CAPTION_ENGINE_KWARGS="enable_prefix_caching=true,disable_log_stats=false"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAPTION_GRPO_CODE=${PROJECT_DIR}/grpo/v51_dvb
export CAPTION_DV_MODE=${CAPTION_DV_MODE:?set CAPTION_DV_MODE}
export CUDA_VISIBLE_DEVICES=$GPUS
export CAPTION_CROP_LAMBDA=$LAMBDA
export CAPTION_STD_FLOOR=0.25
export CAPTION_DV_TABLE=${PROJECT_DIR}/evidence/twobranch_pool_20261020/dv_table.json
export CAPTION_ROLLOUT_LOG=${OUTPUT_DIR}/grpo_v3_A_dv_${TAG}_rollouts.jsonl
POOL=${DATA_DIR}/grpo_caption_v51_v3
MODEL=${OUTPUT_DIR}/sft_v22_turnwise_merged
NAME=grpo_v3_A_dv_$TAG
CKPT=${OUTPUT_DIR}/grpo_v3_A_dv_$TAG
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
    trainer.save_freq=40 trainer.save_model_only=true \
    trainer.save_checkpoint_path=$CKPT \
    "${@:1}"
