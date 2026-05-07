#!/bin/bash
# AlpacaEval — Qwen3-4B 28 个变体
# 移植: 修改 MODEL_ROOT (SDFT 权重根目录), 必要时改 GPU_POOL
# 调用前需 conda activate (lm-eval driver: eval_harness; alpaca driver: 见 helper 顶部注释)

MODEL_ROOT="/ssddata/lihao/projects/SDFT-for-Safe-Alignment/model_weight"
CKPT_SUBDIR="checkpoint-25"
MODEL_BASE="Qwen3-4B"
MODEL_FAMILY="qwen3_4b"

HK_PAIRS=(
    "1 0" "1 2" "1 4" "1 6" "1 8" "1 10" "1 12"
    "2 0"       "2 4" "2 6" "2 8" "2 10" "2 12"
    "4 0"             "4 6" "4 8" "4 10" "4 12"
    "6 0"                   "6 8" "6 10" "6 12"
    "8 0"                         "8 10" "8 12"
    "10 0"                       "10 10" "10 12"
)

GPU_POOL=(4 5 6 7)

source "$(dirname "${BASH_SOURCE[0]}")/_run_0507_alpaca_common.sh"
