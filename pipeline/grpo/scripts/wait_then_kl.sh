#!/bin/bash
# Wait for card-A run (grpo_v3_A_cc, GPU 0,1) to exit; if it completed 165 steps and GPU0/1 are free, launch the KL run there.
cd ${PROJECT_DIR}
LOG=outputs/wait_then_kl.log
echo "$(date) waiting for grpo_v3_A_cc" >> $LOG
while pgrep -f "experiment_name=grpo_v3_A_cc " > /dev/null; do sleep 60; done
sleep 120
STEPS=$(grep -c time_per_step outputs/grpo_v3_A_cc.log)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk "{s+=\$1} END {print s}")
echo "$(date) A exited: steps=$STEPS gpu01_used_mib=$USED" >> $LOG
if [ "$STEPS" -ge 165 ] && [ "$USED" -lt 4000 ]; then
  nohup bash grpo_kl.sh k1e2 0,1 > outputs/grpo_v3_A_kl_k1e2.log 2>&1 &
  echo "$(date) launched grpo_v3_A_kl_k1e2 pid $!" >> $LOG
else
  echo "$(date) NOT launched (A incomplete or GPUs busy)" >> $LOG
fi
