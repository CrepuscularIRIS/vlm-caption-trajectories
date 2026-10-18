#!/bin/bash
# 2+2 双组同跑：A=V2.2/v5.1(GPU0,1,默认Ray) B=v4c/v6(GPU2,3,第二Ray集群)
# 用法: bash grpo_2plus2.sh A|B
set -e
GROUP=$1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=${CACHE_DIR}/hf
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd ${EASYR1_DIR}
export RAY_TMPDIR=/tmp/ray_$GROUP
export CAPTION_ENGINE_KWARGS="enable_prefix_caching=true,disable_log_stats=false"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "$GROUP" = "A" ]; then
  export CAPTION_GRPO_CODE=${PROJECT_DIR}/grpo/v51
  export CUDA_VISIBLE_DEVICES=0,1
  POOL=${DATA_DIR}/grpo_caption_v51_v3
  MODEL=${OUTPUT_DIR}/sft_v22_turnwise_merged
  NAME=grpo_v3_A
  CKPT=${OUTPUT_DIR}/grpo_v3_A
else
  export CAPTION_GRPO_CODE=${PROJECT_DIR}/grpo/v6
  export CUDA_VISIBLE_DEVICES=2,3
  POOL=${DATA_DIR}/grpo_caption_v3
  MODEL=${OUTPUT_DIR}/sft_v4c_merged
  NAME=grpo_v3_B
  CKPT=${OUTPUT_DIR}/grpo_v3_B
fi
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
    "${@:2}"
