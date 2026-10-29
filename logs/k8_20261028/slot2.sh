#!/bin/bash
. "$(dirname "$0")"/common.sh
run k8_grpo $P/grpo_v52m_c/grpo_v52m_c.sh $SFT 2,3 $U2
run k8_gspo $P/grpo_v52m_c/grpo_v52m_c.sh $SFT 2,3 $U2 $GSPO
run k8_base_grpo $P/grpo_v52m_c/grpo_v52m_c.sh $BASE 2,3 $U2
log "slot2 done"
