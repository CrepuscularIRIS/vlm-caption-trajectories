#!/bin/bash
# After the KL run exits on GPU 0,1: E1 of KL@160 (GPU0) then E2 of KL@160 (GPU0); E1 of answer-only GRPO @160 (GPU1).
cd ${PROJECT_DIR}
LOG=outputs/chain_gpu01.log; EZ=${EASYR1_ENV}/bin; E=grpo_eng_repo/eval_v4; OUT=evidence/v3_behavior; P=${PROJECT_DIR}
say() { echo "$(date '+%F %T') $*" >> $LOG; }
ev() {  # gpu code tag model [extra]
  local g=$1 code=$2 tag=$3 model=$4; shift 4
  local sfx=""; [[ "$*" == *first_crop* ]] && sfx="_perturb"
  CAPTION_GRPO_CODE=$code CAPTION_CROP_LAMBDA=0.0 CAPTION_STD_FLOOR=0.25 CAPTION_ROLLOUT_LOG=$OUT/eval_unused_rollouts.jsonl \
    CUDA_VISIBLE_DEVICES=$g $EZ/python $E/eval_v4.py --model $model --code-dir $code --tag $tag --out-dir $OUT "$@" > $OUT/log_${tag}${sfx}.txt 2>&1
  say "eval $tag$sfx exit $?"
}
hf() {
  ls $1/huggingface/*.safetensors > /dev/null 2>&1 || CAPTION_GRPO_CODE=$P/grpo_v51 $EZ/python ${EASYR1_DIR}/scripts/model_merger.py --local_dir $1 >> outputs/merge_after_kl.log 2>&1
  echo $1/huggingface
}
say "evals_after_kl: waiting for grpo_v3_A_kl_k1e2"
while pgrep -f "experiment_name=grpo_v3_A_kl_k1e2 " > /dev/null; do sleep 60; done
sleep 120
while [ ! -d outputs/grpo_v3_AO_base/global_step_160 ]; do sleep 60; done
k=$(hf outputs/grpo_v3_A_kl_k1e2/global_step_160/actor); a=$(hf outputs/grpo_v3_AO_base/global_step_160/actor)
( ev 0 $P/grpo_v51 A_kl160 $k; ev 0 $P/grpo_v51 A_kl160 $k --perturb first_crop ) &
ev 1 $P/grpo_v51_ansonly AO_rl160 $a --pool ${DATA_DIR}/grpo_caption_ao_v3
wait
say "evals_after_kl done; GPU0,1 free"
