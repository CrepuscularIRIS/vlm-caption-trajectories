#!/bin/bash
# GPU 0,1 queue, take 2 (replaces chain_gpu01.sh after its SFT step started): wait for the answer-only SFT ->
# merge -> E1/E2 of card A @160 on GPU0 and E1 of answer-only SFT on GPU1 (in parallel) -> KL GRPO.
cd ${PROJECT_DIR}
LOG=outputs/chain_gpu01.log; M=outputs/sft_v22_answeronly_merged; L=outputs/sft_v22_answeronly_lora
LF=${LF_ENV}/bin; EZ=${EASYR1_ENV}/bin; E=grpo_eng_repo/eval_v4; OUT=evidence/v3_behavior
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "chain_gpu01b: waiting for answer-only SFT process"
while pgrep -f "llamafactory-cli train ${PROJECT_DIR}/sft_v22_answeronly.yaml" > /dev/null; do sleep 30; done
if [ -f $L/adapter_model.safetensors ] && grep -q "train_runtime" outputs/sft_v22_answeronly.log; then
  say "answer-only SFT done"
  CUDA_VISIBLE_DEVICES=0 PATH=$LF:$PATH llamafactory-cli export ${PROJECT_DIR}/merge_answeronly.yaml > outputs/merge_answeronly.log 2>&1
  RC=$?; say "merge exit $RC"
  if [ $RC -eq 0 ]; then
    # Same base tokenizer; LF export writes extra_special_tokens as a list that EasyR1's transformers rejects.
    for f in tokenizer_config.json tokenizer.json chat_template.jinja processor_config.json; do
      [ -f outputs/sft_v22_turnwise_merged/$f ] && cp outputs/sft_v22_turnwise_merged/$f $M/$f
    done
    touch $M/READY; say "merged model READY"
  fi
fi
[ -f $M/READY ] || { mkdir -p $M; touch $M/FAILED; say "answer-only SFT FAILED"; }
(
  CK=outputs/grpo_v3_A_cc/global_step_160/actor
  [ -d $CK/huggingface ] || $EZ/python ${EASYR1_DIR}/scripts/model_merger.py --local_dir $CK > outputs/merge_cc_160.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $EZ/python $E/eval_v4.py --model $CK/huggingface --code-dir ${PROJECT_DIR}/grpo_v51 \
    --tag A_cc160 --out-dir $OUT > $OUT/log_A_cc160.txt 2>&1; echo "E1 A_cc160 exit $?" >> $LOG
  CUDA_VISIBLE_DEVICES=0 $EZ/python $E/eval_v4.py --model $CK/huggingface --code-dir ${PROJECT_DIR}/grpo_v51 \
    --tag A_cc160 --out-dir $OUT --perturb first_crop > $OUT/log_A_cc160_perturb.txt 2>&1; echo "E2 A_cc160 exit $?" >> $LOG
) &
P0=$!
if [ -f $M/READY ]; then
  CUDA_VISIBLE_DEVICES=1 $EZ/python $E/eval_v4.py --model $M --code-dir ${PROJECT_DIR}/grpo/v51_ansonly \
    --pool ${DATA_DIR}/grpo_caption_ao_v3 --tag AO_sft --out-dir $OUT > $OUT/log_AO_sft.txt 2>&1
  say "E1 AO_sft exit $?"
fi
wait $P0
say "evals done; KL GRPO start"
bash grpo_kl.sh k1e2 0,1 > outputs/grpo_v3_A_kl_k1e2.log 2>&1
say "KL GRPO exit $?"
