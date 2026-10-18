#!/bin/bash
# Caption GRPO：双卡（GPU2,3），merged SFT 模型 + 多轮裁图 rollout
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=${CACHE_DIR}/hf
cd ${EASYR1_DIR}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-caption_grpo_$(date +%Y%m%d_%H%M%S)}

CUDA_VISIBLE_DEVICES=2,3 PATH=${EASYR1_ENV}/bin:$PATH python ${PROJECT_DIR}/easyr1_caption.py \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/grpo_caption/train.parquet \
    data.val_files=${DATA_DIR}/grpo_caption/val.parquet \
    data.max_prompt_length=4096 \
    data.max_response_length=7168 \
    data.rollout_batch_size=32 \
    data.val_batch_size=16 \
    data.seed=42 \
    data.filter_overlong_prompts=false \
    data.prompt_key=prompt data.image_key=images data.answer_key=reward_model \
    algorithm.kl_coef=1.0e-3 \
    algorithm.adv_estimator=grpo \
    worker.actor.global_batch_size=32 \
    worker.actor.max_grad_norm=0.1 \
    worker.actor.model.model_path=${OUTPUT_DIR}/sft_vlm_v2_2_merged \
    worker.actor.model.trust_remote_code=true \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.optim.lr_scheduler_type=constant \
    worker.rollout.n=8 \
    worker.rollout.temperature=1.0 \
    worker.rollout.top_p=1.0 \
    worker.rollout.limit_images=10 \
    worker.rollout.gpu_memory_utilization=0.45 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.reward.reward_function=${PROJECT_DIR}/reward_caption.py:compute_score \
    trainer.total_epochs=1 \
    trainer.project_name=caption_grpo \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_freq=20 \
    trainer.save_freq=50 \
    trainer.save_checkpoint_path=${OUTPUT_DIR}/grpo_caption \
    "$@"
