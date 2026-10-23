#!/bin/bash
# Server side (setsid -f). Decision-loss MVE (PREREG.md): unit tests -> train main (beta 1/256) on GPUS_MAIN and control
# (beta 0) on GPUS_CTRL in parallel, 20 updates each -> merge step 20 -> audit (reference states, p scores, recovery)
# and eval_v4 clean/E2 on the 64 pre-specified questions -> dl_compare.py -> DONE.
# Budget rule: the local runner writes STOP_BUDGET after step 1 if the projection exceeds 8 GPU-hours.
set -u
GPUS_MAIN=${1:?gpus main}; GPUS_CTRL=${2:?gpus ctrl}
cd ${PROJECT_DIR}
EZ=${EASYR1_ENV}/bin; P=${PROJECT_DIR}; E=grpo_eng_repo/eval_v4; C=$P/grpo_v52_dl
O=evidence/dl_20261023; S=$O/code; mkdir -p $O
export EASYR1_DIR=${EASYR1_DIR} HF_HOME=${CACHE_DIR}/hf VLLM_ENABLE_V1_MULTIPROCESSING=0 CAPTION_CROP_LAMBDA=0.01
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
R1=outputs/grpo_v3_A_rloo_l010/global_step_40/actor/huggingface
say() { echo "$(date '+%F %T') $*" >> $O/dl.log; }
alive() { pgrep -f "^python .*easyr1_caption.py .*experiment_name=$1 " > /dev/null; }
if [ ! -f $O/unit_ok ]; then
  (cd $C && CAPTION_DL_TOKENIZER=$P/$R1 $EZ/python -m pytest -q -p no:cacheprovider test_decision_loss.py \
     > $P/$O/unit_test.txt 2>&1) && touch $O/unit_ok
  say "unit test $(tail -1 $O/unit_test.txt)"; [ -f $O/unit_ok ] || { touch $O/DONE; exit 2; }
fi
G1=${GPUS_MAIN%%,*}; G2=${GPUS_MAIN##*,}; G3=${GPUS_CTRL%%,*}; G4=${GPUS_CTRL##*,}
if [ ! -f $O/started ]; then touch $O/started
  CAPTION_DL_BETA=0.00390625 setsid nohup bash $S/grpo_dl.sh main 0.01 $GPUS_MAIN trainer.save_limit=-1 \
    > outputs/grpo_v3_A_dl_main.log 2>&1 < /dev/null &
  CAPTION_DL_BETA=0 setsid nohup bash $S/grpo_dl.sh ctrl 0.01 $GPUS_CTRL trainer.save_limit=-1 \
    > outputs/grpo_v3_A_dl_ctrl.log 2>&1 < /dev/null &
  say "started main gpus=$GPUS_MAIN ctrl gpus=$GPUS_CTRL"; sleep 240
fi
while alive grpo_v3_A_dl_main || alive grpo_v3_A_dl_ctrl; do
  if [ -f $O/STOP_BUDGET ]; then
    pkill -f "^python .*easyr1_caption.py .*experiment_name=grpo_v3_A_dl_" ; say "stopped by budget rule"
    echo '{"stopped": "budget"}' > $O/compare_dl.json; touch $O/DONE; exit 0
  fi
  sleep 60
done
say "main steps=$(grep -c time_per_step outputs/grpo_v3_A_dl_main.log) ctrl steps=$(grep -c time_per_step outputs/grpo_v3_A_dl_ctrl.log)"
touch $O/TRAIN_DONE
# revision 4: post-training stages need the protocol code dir (verl/__init__ refuses to fall back)
export CAPTION_GRPO_CODE=$C CAPTION_STD_FLOOR=0.25 CAPTION_DL_H0=12 CAPTION_DL_BETA=0 \
       CAPTION_ROLLOUT_LOG=$P/$O/post_rollouts.jsonl CAPTION_DL_LOG=$P/$O/post_forks.jsonl  # import-time contract only; beta 0 = no aux
hf() { ls $1/huggingface/*.safetensors > /dev/null 2>&1 || CAPTION_GRPO_CODE=$P/grpo_v51 $EZ/python ${EASYR1_DIR}/scripts/model_merger.py --local_dir $1 >> $O/merge.log 2>&1; echo $1/huggingface; }
M=$(hf outputs/grpo_v3_A_dl_main/global_step_20/actor); K=$(hf outputs/grpo_v3_A_dl_ctrl/global_step_20/actor)
for m in $M $K; do [ -f $m/model.safetensors.index.json ] || { say "missing weights $m"; touch $O/DONE; exit 3; }; done
# 64-question pool for eval_v4 (val.parquet = the 64, empty V* file)
$EZ/python - <<PY
import json, pandas as pd
pool = "${DATA_DIR}/grpo_caption_v51_v3"
u = json.load(open("$S/dl_questions.json"))
df = pd.concat([pd.read_parquet(f"{pool}/val.parquet"), pd.read_parquet(f"{pool}/vstar_holdout.parquet")], ignore_index=True)
import os; os.makedirs("$O/pool64", exist_ok=True)
sub = df[df.uid.isin(u)]; assert len(sub) == 64, len(sub)
sub.to_parquet("$O/pool64/val.parquet"); df.iloc[:0].to_parquet("$O/pool64/vstar_holdout.parquet")
PY
AU="$EZ/python $C/dl_audit.py --code-dir $C --questions $S/dl_questions.json"
CUDA_VISIBLE_DEVICES=$G1 $AU --stage states --model $R1 --out $O/audit_states.json >> $O/audit_states.log 2>&1
say "audit states rc=$?"
ev() { local g=$1 tag=$2 m=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g $EZ/python $E/eval_v4.py --model $m --code-dir $C --pool $O/pool64 --samples 2 --tag $tag \
    --out-dir $O "$@" >> $O/eval_$tag.log 2>&1; }
( CUDA_VISIBLE_DEVICES=$G1 $AU --stage score --model $R1 --states-file $O/audit_states.json --out $O/score_ref.json >> $O/audit_score.log 2>&1
  CUDA_VISIBLE_DEVICES=$G1 $AU --stage score --model $M --states-file $O/audit_states.json --out $O/score_main.json >> $O/audit_score.log 2>&1
  CUDA_VISIBLE_DEVICES=$G1 $AU --stage score --model $K --states-file $O/audit_states.json --out $O/score_ctrl.json >> $O/audit_score.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=$G2 $AU --stage recover --model $M --states-file $O/audit_states.json --out $O/recover_main.json >> $O/audit_recover.log 2>&1
  CUDA_VISIBLE_DEVICES=$G2 $AU --stage recover --model $K --states-file $O/audit_states.json --out $O/recover_ctrl.json >> $O/audit_recover.log 2>&1 ) &
( ev $G3 main $M; ev $G3 main $M --perturb first_crop ) &
( ev $G4 ctrl $K; ev $G4 ctrl $K --perturb first_crop ) &
wait
say "audit + eval done"
$EZ/python $S/dl_compare.py $O outputs $O/compare_dl.json > $O/compare_stdout.txt 2>&1
say "compare rc=$?"
touch $O/DONE
