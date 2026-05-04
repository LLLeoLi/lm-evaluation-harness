#!/bin/bash
# ==============================================================================
# setup.sh — 安装 lm-evaluation-harness 通用能力评测所需的依赖
#
# 使用方法:
#   cd /opt/tiger/entry/lm-evaluation-harness
#   bash setup.sh
# ==============================================================================

set -e

# ----------- 路径配置 ----------------------------------------------------------
ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"

echo "=================================================="
echo "[setup] ENTRY_DIR = ${ENTRY_DIR}"
echo "[setup] LME_DIR   = ${LME_DIR}"
echo "=================================================="

mkdir -p "${ENTRY_DIR}"
cd "${LME_DIR}"

# ----------- 1. lm-evaluation-harness (editable) + lm_eval[hf] ----------------
echo "[setup] 安装 lm-evaluation-harness ..."
pip install -e . --user
pip install "lm_eval[hf]" --user
