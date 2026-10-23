#!/bin/bash
# Server side (setsid nohup). Dose check of the mentor's formula r = correct*(1 - lambda*crops) under RLOO (no std),
# lambda 0.05 (GPU 0,1) and 0.10 (GPU 2,3); everything else identical to r0/r1 (grpo_rloo.sh, same SFT start, seed,
# data order, 165 steps). Stage A showed the lambda=0.01 term is only 2-5% of sum|A|. Merges steps 40/80/120/160,
# runs E1 + E2 (first crop shifted) per checkpoint on 4 GPUs, touches DONE.
set -u
cd ${PROJECT_DIR}
O=evidence/lsweep_20261023; R=evidence/rloo_20261021/code; mkdir -p $O
EZ=${EASYR1_ENV}/bin; P=${PROJECT_DIR}  # [scrub]; E=grpo_eng_repo/eval_v4
say() { echo "$(date '+%F %T') $*" >> $O/lsweep.log; }
alive() { pgrep -f "^python .*easyr1_caption.py .*experiment_name=$1 " > /dev/null; }
start() { [ -f $O/started_$1 ] && return; touch $O/started_$1
  setsid nohup bash $R/grpo_rloo.sh $1 $2 $3 trainer.save_limit=-1 > outputs/grpo_v3_A_rloo_$1.log 2>&1 < /dev/null &
  say "started $1 lambda=$2 gpus=$3"; }
start l050 0.05 0,1
start l100 0.10 2,3
sleep 180
while alive grpo_v3_A_rloo_l050 || alive grpo_v3_A_rloo_l100; do sleep 60; done
for t in l050 l100; do n=$(grep -c time_per_step outputs/grpo_v3_A_rloo_$t.log); say "$t steps=$n"; [ "$n" -ge 165 ] || { say "INCOMPLETE $t"; exit 5; }; done
touch $O/TRAIN_DONE
hf() { ls $1/huggingface/*.safetensors > /dev/null 2>&1 || CAPTION_GRPO_CODE=$P/grpo_v51 $EZ/python ${EASYR1_DIR}/scripts/model_merger.py --local_dir $1 >> $O/merge.log 2>&1; echo $1/huggingface; }
ev() { local g=$1 tag=$2 m=$3; shift 3
  CAPTION_GRPO_CODE=$P/grpo_v51 CUDA_VISIBLE_DEVICES=$g $EZ/python $E/eval_v4.py --model $m --code-dir $P/grpo_v51 --tag $tag --out-dir $O "$@" >> $O/eval_$tag.log 2>&1; }
for st in 40 80 120 160; do
  a=$(hf outputs/grpo_v3_A_rloo_l050/global_step_$st/actor); b=$(hf outputs/grpo_v3_A_rloo_l100/global_step_$st/actor)
  ev 0 r5_$st $a & ev 1 r10_$st $b & ev 2 r5_$st $a --perturb first_crop & ev 3 r10_$st $b --perturb first_crop & wait
done
say "eval done"
touch $O/DONE
