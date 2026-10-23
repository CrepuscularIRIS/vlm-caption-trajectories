#!/bin/bash
# Server side (setsid -f). Mentor 10-21 four-case tool credit, one arm vs the existing RLOO controls:
#  1 unit test (CPU; must pass)
#  2 train fc = CAPTION_FC_MODE=fourcase, r1's recipe (RLOO, λ 0.01, seed 42, same data order), 165 steps, GPU 2,3,
#    CAPTION_FORK_M=4 forks per follow-up state, CAPTION_FORK_M0=16 shared forks at s_1
#  3 merge steps 80 / 160, E1 + E2 (GPU 2 / 3), compare_fc.py against evidence/rloo_20261021 (r1, r0), DONE.
set -u
cd ${PROJECT_DIR}
O=evidence/fc_20261022; S=$O/code; mkdir -p $O
EZ=${EASYR1_ENV}/bin; P=${PROJECT_DIR}; E=grpo_eng_repo/eval_v4
export EASYR1_DIR=${EASYR1_DIR} HF_HOME=${CACHE_DIR}/hf VLLM_ENABLE_V1_MULTIPROCESSING=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
say() { echo "$(date '+%F %T') $*" >> $O/fc.log; }
alive() { pgrep -f "^python .*easyr1_caption.py .*experiment_name=$1 " > /dev/null; }
if [ ! -f $O/unit_ok ]; then
  (cd $P/grpo_v51_fc && $EZ/python -m pytest -q -p no:cacheprovider test_fourcase_credit.py > $P/$O/unit_test.txt 2>&1) && touch $O/unit_ok
  say "unit test $(tail -1 $O/unit_test.txt)"; [ -f $O/unit_ok ] || { touch $O/DONE; exit 2; }
fi
if [ ! -f $O/started_fc ]; then touch $O/started_fc
  CAPTION_FC_MODE=fourcase CAPTION_FORK_M=4 CAPTION_FORK_M0=16 setsid nohup bash $S/grpo_fc.sh fc 0.01 2,3 trainer.save_limit=-1 \
    > outputs/grpo_v3_A_fc_fc.log 2>&1 < /dev/null &
  say "started fc gpus=2,3"; sleep 240
fi
while alive grpo_v3_A_fc_fc; do sleep 60; done
say "fc steps=$(grep -c time_per_step outputs/grpo_v3_A_fc_fc.log)"
[ -f $O/STOP_BUDGET ] && { say "stopped by budget rule"; echo '{"stopped": "budget"}' > $O/compare_fc.json; touch $O/DONE; exit 0; }
touch $O/TRAIN_DONE
hf() { ls $1/huggingface/*.safetensors > /dev/null 2>&1 || CAPTION_GRPO_CODE=$P/grpo_v51 $EZ/python ${EASYR1_DIR}/scripts/model_merger.py --local_dir $1 >> $O/merge.log 2>&1; echo $1/huggingface; }
ev() { local g=$1 tag=$2 m=$3; shift 3
  [ -f $m/model.safetensors.index.json ] || { say "skip eval $tag (no weights)"; return; }
  CAPTION_GRPO_CODE=$P/grpo_v51 CUDA_VISIBLE_DEVICES=$g $EZ/python $E/eval_v4.py --model $m --code-dir $P/grpo_v51 --tag $tag --out-dir $O "$@" >> $O/eval_$tag.log 2>&1; }
for st in 80 160; do
  [ -d outputs/grpo_v3_A_fc_fc/global_step_$st ] || { say "no ckpt $st"; continue; }
  a=$(hf outputs/grpo_v3_A_fc_fc/global_step_$st/actor)
  ev 2 fc$st $a & ev 3 fc$st $a --perturb first_crop & wait
  say "eval $st done"
done
$EZ/python $S/compare_fc.py $O evidence/rloo_20261021 outputs $O/compare_fc.json > $O/compare_stdout.txt 2>&1
say "compare rc=$?"
touch $O/DONE
