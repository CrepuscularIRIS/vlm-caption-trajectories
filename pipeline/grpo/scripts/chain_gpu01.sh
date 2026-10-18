#!/bin/bash
# GPU 0,1 queue: wait for card A (grpo_v3_A_cc) -> answer-only SFT + merge (mentor baseline) -> KL GRPO.
cd ${PROJECT_DIR}
LOG=outputs/chain_gpu01.log; M=outputs/sft_v22_answeronly_merged
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "waiting for grpo_v3_A_cc"
while pgrep -f "experiment_name=grpo_v3_A_cc " > /dev/null; do sleep 60; done
sleep 120
STEPS=$(grep -c time_per_step outputs/grpo_v3_A_cc.log)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk '{s+=$1} END {print s}')
say "A exited: steps=$STEPS gpu01_used_mib=$USED"
if [ "$STEPS" -lt 165 ] || [ "$USED" -ge 4000 ]; then say "STOP: A incomplete or GPUs busy"; exit 1; fi
say "answer-only SFT start"
( cd ${DATA_DIR}/vlm-sft-v2-2 && CUDA_VISIBLE_DEVICES=0,1 PATH=${LF_ENV}/bin:$PATH FORCE_TORCHRUN=1 \
  llamafactory-cli train ${PROJECT_DIR}/sft_v22_answeronly.yaml ) > outputs/sft_v22_answeronly.log 2>&1
RC=$?; say "answer-only SFT exit $RC"
if [ $RC -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 PATH=${LF_ENV}/bin:$PATH llamafactory-cli export ${PROJECT_DIR}/merge_answeronly.yaml > outputs/merge_answeronly.log 2>&1
  RC=$?; say "merge exit $RC"
  if [ $RC -eq 0 ]; then
    # Same base tokenizer; LF export writes extra_special_tokens as a list that EasyR1's transformers rejects.
    for f in tokenizer_config.json tokenizer.json chat_template.jinja processor_config.json; do
      [ -f outputs/sft_v22_turnwise_merged/$f ] && cp outputs/sft_v22_turnwise_merged/$f $M/$f
    done
    touch $M/READY; say "merged model READY"
  fi
fi
[ -f $M/READY ] || { mkdir -p $M; touch $M/FAILED; say "answer-only SFT FAILED; continuing with KL"; }
say "KL GRPO start"
bash grpo_kl.sh k1e2 0,1 > outputs/grpo_v3_A_kl_k1e2.log 2>&1
say "KL GRPO exit $?"
