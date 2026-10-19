#!/bin/bash
# GPU 0,1 queue, take 3: E1/E2 evals (CAPTION_GRPO_CODE set per protocol; take 2 forgot it) -> KL GRPO.
# GPU0: card A @160 E1 + E2. GPU1: answer-only SFT E1, then card A @165 E1.
cd ${PROJECT_DIR}
LOG=outputs/chain_gpu01.log; M=outputs/sft_v22_answeronly_merged
EZ=${EASYR1_ENV}/bin; E=grpo_eng_repo/eval_v4; OUT=evidence/v3_behavior; P=${PROJECT_DIR}
say() { echo "$(date '+%F %T') $*" >> $LOG; }
ev() {  # gpu code_dir tag model [extra args]
  local g=$1 code=$2 tag=$3 model=$4; shift 4
  local sfx=""; [[ "$*" == *first_crop* ]] && sfx="_perturb"
  CAPTION_GRPO_CODE=$code CUDA_VISIBLE_DEVICES=$g $EZ/python $E/eval_v4.py --model $model --code-dir $code \
    --tag $tag --out-dir $OUT "$@" > $OUT/log_${tag}${sfx}.txt 2>&1
  say "eval $tag$sfx exit $?"
}
hf() {  # FSDP actor dir -> huggingface dir
  [ -d $1/huggingface ] || CAPTION_GRPO_CODE=$P/grpo_v51 $EZ/python ${EASYR1_DIR}/scripts/model_merger.py --local_dir $1 >> outputs/merge_cc.log 2>&1
  echo $1/huggingface
}
say "chain_gpu01c: evals start"
( m=$(hf outputs/grpo_v3_A_cc/global_step_160/actor)
  ev 0 $P/grpo_v51 A_cc160 $m
  ev 0 $P/grpo_v51 A_cc160 $m --perturb first_crop ) &
P0=$!
( ev 1 $P/grpo_v51_ansonly AO_sft $M --pool ${DATA_DIR}/grpo_caption_ao_v3
  m=$(hf outputs/grpo_v3_A_cc/global_step_165/actor)
  ev 1 $P/grpo_v51 A_cc165 $m ) &
P1=$!
wait $P0 $P1
say "evals done; KL GRPO start"
bash grpo_kl.sh k1e2 0,1 > outputs/grpo_v3_A_kl_k1e2.log 2>&1
say "KL GRPO exit $?"
