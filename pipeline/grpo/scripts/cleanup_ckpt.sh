#!/bin/bash
# Keep only merged HF weights at step 160 (evaluated) and step 165 (final) of finished GRPO runs; delete intermediate
# checkpoints and FSDP shards. Never touches the running control run or any non-GRPO directory.
cd ${OUTPUT_DIR}
LOG=../logs/cleanup_ckpt_20261019.txt; EZ=${EASYR1_ENV}/bin
say() { echo "$(date '+%F %T') $*" >> $LOG; }
hfok() { ls $1/actor/huggingface/*.safetensors > /dev/null 2>&1; }
say "start; free=$(df -h ${DATA_ROOT} | tail -1 | awk '{print $4}')"
for r in grpo_v3_A_cc grpo_v3_A_cc_dsasym grpo_v3_A_floor_l025 grpo_v3_A_floor_l050 grpo_v3_A_kl_k1e2 grpo_v3_A_cropcost grpo_v3_AO_base; do
  for s in $r/global_step_*; do
    n=${s##*_}
    if [ "$n" = 160 ] || [ "$n" = 165 ] || { [ "$r" = grpo_v3_A_cropcost ] && [ "$n" = 40 ]; }; then
      if ! hfok $s; then
        CAPTION_GRPO_CODE=${PROJECT_DIR}/grpo_v51 $EZ/python ${EASYR1_DIR}/scripts/model_merger.py --local_dir $s/actor >> ../logs/cleanup_merge_20261019.log 2>&1
        say "merged $s rc=$?"
      fi
      if hfok $s; then
        for f in $s/actor/model_world_size_*.pt $s/actor/extra_state_world_size_*.pt; do [ -f $f ] && { say "rm $(du -h $f | cut -f1) $f"; rm -f $f; }; done
      else
        say "KEEP shards (merge failed) $s"
      fi
    else
      say "rm -r $(du -sh $s | cut -f1) $s"; rm -rf $s
    fi
  done
done
say "done; free=$(df -h ${DATA_ROOT} | tail -1 | awk '{print $4}')"
