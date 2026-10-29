#!/bin/bash
. "$(dirname "$0")"/common.sh
run k8_dapo $P/grpo_v52m_c/grpo_v52m_c.sh $SFT 0,1 $U1 $DAPO
CAPTION_GIGPO=1 run k8_gigpo $P/grpo_gigpo/grpo_gigpo.sh $SFT 0,1 $U2
log "slot1 done"
