#!/bin/bash
# B6: base Qwen3-VL-4B + GRPO WITHOUT tool. Plain EasyR1 single-turn (no caption adapter).
# Same questions/images as grpo_v52 (8k pool -> direct prompt = eval_base direct mode), same recipe:
# 128 q x 16, mini 32 (4 updates/step), 200 steps, max_grad_norm 1.0, no KL, correctness reward only.
# Response budget 4096 (B0 direct used ~460 tokens on average). Full checkpoints, last 2 kept.
# usage: bash grpo_v51fix_notool.sh TAG MODEL_PATH GPUS [extra overrides...]
set -e
TAG=$1; MODEL=$2; GPUS=$3
shift 3 2>/dev/null || true
[ -n "$TAG" ] && [ -n "$MODEL" ] && [ -n "$GPUS" ] || { echo "usage: $0 TAG MODEL GPUS"; exit 2; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=${CACHE_DIR}/hf
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd ${EASYR1_DIR}
export RAY_TMPDIR=/tmp/ray_v52_$TAG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAPTION_GRPO_CODE=${PROJECT_DIR}/grpo/v51fix/notool  # no-op patch: stock EasyR1 rollout
export CUDA_VISIBLE_DEVICES=$GPUS
export AUDIT_SESSION_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
OUT=${PROJECT_DIR}/outputs/v52_$TAG
mkdir -p $OUT
export CAPTION_ROLLOUT_LOG=$OUT/rollouts.jsonl
DATA=${DATA_DIR}/grpo_direct_v51_8k_fullq_codex
NGPU=$(echo $GPUS | tr ',' '\n' | wc -l)
export PATH=${EASYR1_ENV}/bin:$PATH
python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=$DATA/train.parquet data.val_files=$DATA/val.parquet \
    data.prompt_key=prompt data.image_key=images data.answer_key=answer data.format_prompt=null \
    data.max_prompt_length=4096 data.max_response_length=4096 \
    data.max_pixels=589824 data.min_pixels=1024 \
    data.rollout_batch_size=128 data.val_batch_size=192 data.seed=42 \
    data.filter_overlong_prompts=false \
    algorithm.adv_estimator=grpo algorithm.kl_coef=0.0 algorithm.disable_kl=true \
    algorithm.online_filtering=false \
    worker.actor.global_batch_size=32 worker.actor.max_grad_norm=1.0 \
    worker.actor.clip_ratio_low=0.2 worker.actor.clip_ratio_high=0.3 \
    worker.actor.model.model_path=$MODEL worker.actor.model.trust_remote_code=true \
    worker.actor.optim.lr=1.0e-6 worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.optim.lr_scheduler_type=constant \
    worker.actor.on_policy_reuse_log_probs=false \
    worker.actor.offload.offload_params=false worker.actor.offload.offload_optimizer=false \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.rollout.n=16 worker.rollout.temperature=0.7 worker.rollout.top_p=1.0 \
    worker.rollout.gpu_memory_utilization=0.45 worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${PROJECT_DIR}/grpo/v51fix/reward_direct.py:compute_score \
    trainer.total_epochs=4 trainer.max_steps=200 \
    trainer.project_name=caption_v52 trainer.experiment_name=v52_$TAG \
    trainer.logger='["console"]' trainer.n_gpus_per_node=$NGPU trainer.nnodes=1 \
    trainer.val_freq=50 trainer.val_generations_to_log=0 \
    trainer.save_freq=50 trainer.save_limit=2 trainer.save_model_only=false \
    trainer.save_checkpoint_path=$OUT/ckpt \
    "${@:1}"
