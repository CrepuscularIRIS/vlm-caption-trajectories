#!/bin/bash
# GPU 2,3 queue: wait for card B (grpo_v3_A_cc_dsasym) -> answer-only GRPO (needs merged SFT from chain_gpu01) -> logged no-KL control.
cd ${PROJECT_DIR}
LOG=outputs/chain_gpu23.log; M=outputs/sft_v22_answeronly_merged
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "waiting for grpo_v3_A_cc_dsasym"
while pgrep -f "experiment_name=grpo_v3_A_cc_dsasym " > /dev/null; do sleep 60; done
sleep 120
STEPS=$(grep -c time_per_step outputs/grpo_v3_A_cc_dsasym.log)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END {print s}')
say "B exited: steps=$STEPS gpu23_used_mib=$USED"
if [ "$STEPS" -lt 165 ] || [ "$USED" -ge 4000 ]; then say "STOP: B incomplete or GPUs busy"; exit 1; fi
while [ ! -f $M/READY ] && [ ! -f $M/FAILED ]; do sleep 60; done
if [ -f $M/READY ]; then
  say "answer-only GRPO start"
  bash grpo_ao.sh base 2,3 > outputs/grpo_v3_AO_base.log 2>&1
  say "answer-only GRPO exit $?"
else
  say "answer-only SFT failed; skipping answer-only GRPO"
fi
say "control rerun start"
bash grpo_cropfloor.sh ctrl_l000 0.0 2,3 > outputs/grpo_v3_A_floor_ctrl_l000.log 2>&1
say "control rerun exit $?"
