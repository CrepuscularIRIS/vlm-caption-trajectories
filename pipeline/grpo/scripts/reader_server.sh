#!/bin/bash
# Frozen reader R for the four-case tool credit: Qwen3-VL-4B-Instruct (base, never trained) behind vLLM's OpenAI API.
# Usage: bash reader_server.sh <GPU> <PORT>; logs to outputs/reader_server_<PORT>.log
GPU=$1; PORT=$2
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=${CACHE_DIR}/hf
CUDA_VISIBLE_DEVICES=$GPU exec ${EASYR1_ENV}/bin/vllm serve ${MODEL_DIR}/Qwen3-VL-4B-Instruct \
  --served-model-name reader --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 11264 \
  --limit-mm-per-prompt '{"image": 10}' --gpu-memory-utilization 0.85 --max-logprobs 20 --seed 0
