#!/bin/bash
# Mentor 10-22 first-action decision loss MVE (Pro design 10-22; PREREG.md). Both arms: protocol v5.2i, r1's recipe
# (RLOO, r = correct*(1-λ*crops), λ 0.01, seed 42), started from r1 step-40 HF weights with r1's step-40 dataloader
# state (questions 41-60), fresh optimizer, 20 updates. main: CAPTION_DL_BETA=0.00390625 (1/256), K=4+4 forks at one
# pre-assigned slot per question; control: CAPTION_DL_BETA=0 (no forks, no aux loss).
# 用法: CAPTION_DL_BETA=<beta> bash grpo_dl.sh <TAG> <LAMBDA> <GPUS> [overrides]

set -e
TAG=$1; LAMBDA=$2; GPUS=$3
shift 3 2>/dev/null || true
[ -n "$TAG" ] && [ -n "$LAMBDA" ] && [ -n "$GPUS" ] || { echo "usage: $0 TAG LAMBDA GPUS"; exit 2; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=${CACHE_DIR}/hf
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd ${EASYR1_DIR}
export RAY_TMPDIR=/tmp/ray_dl_$TAG
export CAPTION_ENGINE_KWARGS="enable_prefix_caching=true,disable_log_stats=false"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAPTION_GRPO_CODE=${PROJECT_DIR}/grpo_v52_dl
export CAPTION_DL_BETA=${CAPTION_DL_BETA:?set CAPTION_DL_BETA}
export CAPTION_DL_K=4 CAPTION_DL_H0=12
R1=${OUTPUT_DIR}/grpo_v3_A_rloo_l010/global_step_40
export CAPTION_DL_DATALOADER=$R1/dataloader.pt
export CUDA_VISIBLE_DEVICES=$GPUS
export CAPTION_CROP_LAMBDA=$LAMBDA
export CAPTION_STD_FLOOR=0.25
export CAPTION_ROLLOUT_LOG=${OUTPUT_DIR}/grpo_v3_A_dl_${TAG}_rollouts.jsonl
export CAPTION_DL_LOG=${OUTPUT_DIR}/grpo_v3_A_dl_${TAG}_forks.jsonl
POOL=${DATA_DIR}/grpo_caption_v51_v3
MODEL=$R1/actor/huggingface
NAME=grpo_v3_A_dl_$TAG
# PREREG revision 1 (before launch): EasyR1 warmup = int(0.05 * 20) = 1 step with lr(step 0) = 0/1 = 0, which would
# waste update 1; r1 at its step 41 runs at constant 1e-6, so both arms use no warmup (worker.actor.optim.lr_warmup_ratio=0).
CKPT=${OUTPUT_DIR}/grpo_v3_A_dl_$TAG
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
    algorithm.adv_estimator=rloo algorithm.kl_coef=0.0 algorithm.disable_kl=true \
    algorithm.online_filtering=false \
    worker.actor.global_batch_size=12 worker.actor.max_grad_norm=0.1 \
    worker.actor.model.model_path=$MODEL worker.actor.model.trust_remote_code=true \
    worker.actor.optim.lr=1.0e-6 worker.actor.optim.lr_warmup_ratio=0.0 \
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
    trainer.total_epochs=1 trainer.max_steps=20 \
    trainer.project_name=caption_dl trainer.experiment_name=$NAME \
    trainer.logger='["console"]' trainer.n_gpus_per_node=2 trainer.nnodes=1 \
    trainer.val_freq=-1 trainer.val_before_train=false trainer.val_generations_to_log=0 \
    trainer.save_freq=20 trainer.save_model_only=true \
    trainer.save_checkpoint_path=$CKPT \
    "${@:1}"
