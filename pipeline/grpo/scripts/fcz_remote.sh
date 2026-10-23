#!/bin/bash
# Server side (setsid -f). Control arm for t-fourcase-rl: CAPTION_FC_MODE=zero (crop turns get -λ, no before/after
# information; same forks, same rewrite condition), r1's recipe, 165 steps on GPU 0,1; then E1 + E2 at 80 / 160.
set -u
cd ${PROJECT_DIR}
O=evidence/fcz_20261022; S=$O/code; mkdir -p $O
EZ=${EASYR1_ENV}/bin; P=${PROJECT_DIR}; E=grpo_eng_repo/eval_v4
export EASYR1_DIR=${EASYR1_DIR} HF_HOME=${CACHE_DIR}/hf VLLM_ENABLE_V1_MULTIPROCESSING=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
say() { echo "$(date '+%F %T') $*" >> $O/fcz.log; }
alive() { pgrep -f "^python .*easyr1_caption.py .*experiment_name=$1 " > /dev/null; }
if [ ! -f $O/unit_ok ]; then
  (cd $P/grpo_v51_fcz && $EZ/python -m pytest -q -p no:cacheprovider test_fourcase_credit.py > $P/$O/unit_test.txt 2>&1) && touch $O/unit_ok
  say "unit test $(tail -1 $O/unit_test.txt)"; [ -f $O/unit_ok ] || { touch $O/DONE; exit 2; }
fi
if [ ! -f $O/started_zero ]; then touch $O/started_zero
  CAPTION_FC_MODE=zero CAPTION_FORK_M=4 CAPTION_FORK_M0=16 setsid nohup bash $S/grpo_fcz.sh zero 0.01 0,1 trainer.save_limit=-1 \
    > outputs/grpo_v3_A_fc_zero.log 2>&1 < /dev/null &
  say "started zero gpus=0,1"; sleep 240
fi
while alive grpo_v3_A_fc_zero; do sleep 60; done
say "zero steps=$(grep -c time_per_step outputs/grpo_v3_A_fc_zero.log)"
touch $O/TRAIN_DONE
hf() { ls $1/huggingface/*.safetensors > /dev/null 2>&1 || CAPTION_GRPO_CODE=$P/grpo_v51 $EZ/python ${EASYR1_DIR}/scripts/model_merger.py --local_dir $1 >> $O/merge.log 2>&1; echo $1/huggingface; }
ev() { local g=$1 tag=$2 m=$3; shift 3
  [ -f $m/model.safetensors.index.json ] || { say "skip eval $tag (no weights)"; return; }
  CAPTION_GRPO_CODE=$P/grpo_v51 CUDA_VISIBLE_DEVICES=$g $EZ/python $E/eval_v4.py --model $m --code-dir $P/grpo_v51 --tag $tag --out-dir $O "$@" >> $O/eval_$tag.log 2>&1; }
for st in 80 160; do
  [ -d outputs/grpo_v3_A_fc_zero/global_step_$st ] || { say "no ckpt $st"; continue; }
  a=$(hf outputs/grpo_v3_A_fc_zero/global_step_$st/actor)
  ev 0 zero$st $a & ev 1 zero$st $a --perturb first_crop & wait
  say "eval $st done"
done
touch $O/DONE
