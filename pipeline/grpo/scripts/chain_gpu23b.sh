#!/bin/bash
# GPU 2,3 queue, take 2: wait for card B (grpo_v3_A_cc_dsasym) -> E1/E2 of B (GPU2: E1 @160 then E1 @165; GPU3: E2 @160)
# -> answer-only GRPO (mentor baseline) -> logged no-KL control.
cd ${PROJECT_DIR}
LOG=outputs/chain_gpu23.log; M=outputs/sft_v22_answeronly_merged
EZ=${EASYR1_ENV}/bin; E=grpo_eng_repo/eval_v4; OUT=evidence/v3_behavior; P=${PROJECT_DIR}
say() { echo "$(date '+%F %T') $*" >> $LOG; }
ev() {  # gpu tag model [extra]
  local g=$1 tag=$2 model=$3; shift 3
  local sfx=""; [[ "$*" == *first_crop* ]] && sfx="_perturb"
  CAPTION_GRPO_CODE=$P/grpo_v51 CUDA_VISIBLE_DEVICES=$g $EZ/python $E/eval_v4.py --model $model --code-dir $P/grpo_v51 \
    --tag $tag --out-dir $OUT "$@" > $OUT/log_${tag}${sfx}.txt 2>&1
  say "eval $tag$sfx exit $?"
}
hf() {
  ls $1/huggingface/*.safetensors > /dev/null 2>&1 || CAPTION_GRPO_CODE=$P/grpo_v51 $EZ/python ${EASYR1_DIR}/scripts/model_merger.py --local_dir $1 >> outputs/merge_ccb.log 2>&1
  echo $1/huggingface
}
say "chain_gpu23b: waiting for grpo_v3_A_cc_dsasym"
while pgrep -f "experiment_name=grpo_v3_A_cc_dsasym " > /dev/null; do sleep 60; done
sleep 120
STEPS=$(grep -c time_per_step outputs/grpo_v3_A_cc_dsasym.log)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END {print s}')
say "B exited: steps=$STEPS gpu23_used_mib=$USED"
if [ "$STEPS" -lt 165 ] || [ "$USED" -ge 4000 ]; then say "STOP: B incomplete or GPUs busy"; exit 1; fi
m160=$(hf outputs/grpo_v3_A_cc_dsasym/global_step_160/actor)
( ev 2 A_ccb160 $m160; m165=$(hf outputs/grpo_v3_A_cc_dsasym/global_step_165/actor); ev 2 A_ccb165 $m165 ) &
P2=$!
ev 3 A_ccb160 $m160 --perturb first_crop
wait $P2
if [ -f $M/READY ]; then
  say "answer-only GRPO start"
  bash grpo_ao.sh base 2,3 > outputs/grpo_v3_AO_base.log 2>&1
  say "answer-only GRPO exit $?"
fi
say "control rerun start"
bash grpo_cropfloor.sh ctrl_l000 0.0 2,3 > outputs/grpo_v3_A_floor_ctrl_l000.log 2>&1
say "control rerun exit $?"
