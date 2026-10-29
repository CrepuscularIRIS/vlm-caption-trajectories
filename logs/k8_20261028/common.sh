# 4k x 8 formal runs (user 10-28; plan obs/notes/mainline/14): pool grpo_caption_v51_4k (4,096 q, unique original images),
# n = 8, 32 steps x 128 q, actor mini-batch 32 q (4 updates/step), lr 1e-6 const, grad norm 1.0, no KL, caption v5.1,
# 0/1 reward, single seed; grpo_v52m_c lazy + per-view cache 16G + prefetch 16 (window 128); val at 0/25/32; save every 5, keep 2.
P=${PROJECT_DIR}; E=$P/evidence/k8_20261028
SFT=$P/outputs/sft_v22_turnwise_merged; BASE=${MODEL_DIR}/Qwen3-VL-4B-Instruct
export TRAIN=${DATA_DIR}/grpo_caption_v51_4k/train.parquet ROLL_N=8 LAZY=1 MINI=32
export CAPTION_MM_CACHE_GB=16 CAPTION_MM_PREFETCH=16 CAPTION_MM_PREFETCH_WINDOW=128
COMMON="data.val_files=${DATA_DIR}/grpo_caption_v51_4k/val_nosft.parquet trainer.max_steps=32 trainer.save_freq=5 trainer.save_limit=2"
U2="worker.actor.micro_batch_size_per_device_for_update=2 worker.actor.micro_batch_size_per_device_for_experience=4"
U1="worker.actor.micro_batch_size_per_device_for_update=1 worker.actor.micro_batch_size_per_device_for_experience=4"
DAPO="algorithm.online_filtering=true algorithm.filter_key=overall algorithm.filter_low=0.01 algorithm.filter_high=0.99 trainer.max_try_make_batch=100 worker.actor.clip_ratio_low=0.2 worker.actor.clip_ratio_high=0.28 worker.actor.loss_avg_mode=token"
GSPO="worker.actor.loss_type=gspo worker.actor.loss_avg_mode=seq worker.actor.clip_ratio_low=0.0003 worker.actor.clip_ratio_high=0.0004"
log() { echo "$(date "+%F %T") $*" >> $E/launch.log; }
run() { local tag=$1 launcher=$2 model=$3 gpus=$4; shift 4; log "start $tag gpus=$gpus model=$(basename $(dirname $model))/$(basename $model)"
  bash $launcher $tag $model $gpus $COMMON "$@" > $E/$tag.log 2>&1; log "$tag exit=$?"; }
