#!/bin/bash
# Wait for card-B run (grpo_v3_A_cc_dsasym, GPU 2,3) to exit; if it completed 165 steps and GPU2/3 are free, launch the no-KL logged control (original GRPO-A config) there.
cd ${PROJECT_DIR}
LOG=outputs/wait_then_ctrl.log
echo "$(date) waiting for grpo_v3_A_cc_dsasym" >> $LOG
while pgrep -f "experiment_name=grpo_v3_A_cc_dsasym " > /dev/null; do sleep 60; done
sleep 120
STEPS=$(grep -c time_per_step outputs/grpo_v3_A_cc_dsasym.log)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk "{s+=\$1} END {print s}")
echo "$(date) A exited: steps=$STEPS gpu23_used_mib=$USED" >> $LOG
if [ "$STEPS" -ge 165 ] && [ "$USED" -lt 4000 ]; then
  nohup bash grpo_cropfloor.sh ctrl_l000 0.0 2,3 > outputs/grpo_v3_A_floor_ctrl_l000.log 2>&1 &
  echo "$(date) launched grpo_v3_A_floor_ctrl_l000 pid $!" >> $LOG
else
  echo "$(date) NOT launched (A incomplete or GPUs busy)" >> $LOG
fi
