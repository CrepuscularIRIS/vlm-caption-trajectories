#!/bin/bash
# GPU 2,3 queue, take 3: re-run answer-only SFT E1 with the fixed config.json (rope_scaling restored)
# -> answer-only GRPO -> logged no-KL control.
cd ${PROJECT_DIR}
LOG=outputs/chain_gpu23.log; M=outputs/sft_v22_answeronly_merged
EZ=${EASYR1_ENV}/bin; OUT=evidence/v3_behavior; P=${PROJECT_DIR}
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "chain_gpu23c: AO_sft E1 rerun"
CAPTION_GRPO_CODE=$P/grpo_v51_ansonly CAPTION_CROP_LAMBDA=0.0 CAPTION_STD_FLOOR=0.25 CAPTION_ROLLOUT_LOG=$OUT/eval_unused_rollouts.jsonl \
  CUDA_VISIBLE_DEVICES=2 $EZ/python grpo_eng_repo/eval_v4/eval_v4.py --model $M --code-dir $P/grpo_v51_ansonly \
  --pool ${DATA_DIR}/grpo_caption_ao_v3 --tag AO_sft --out-dir $OUT > $OUT/log_AO_sft.txt 2>&1
say "eval AO_sft exit $?"
say "answer-only GRPO start"
bash grpo_ao.sh base 2,3 > outputs/grpo_v3_AO_base.log 2>&1
say "answer-only GRPO exit $?"
say "control rerun start"
bash grpo_cropfloor.sh ctrl_l000 0.0 2,3 > outputs/grpo_v3_A_floor_ctrl_l000.log 2>&1
say "control rerun exit $?"
