#!/bin/bash
# ==========================================
# 环境安装脚本（迁移到 /opt/tiger/entry 后使用）
#
# 前置：项目代码已放到 /opt/tiger/entry 下，
#       lm-evaluation-harness/ 在 /opt/tiger/entry/lm-evaluation-harness
#
# 用法：
#   bash /opt/tiger/entry/lm-evaluation-harness/setup.sh
# ==========================================

set -e

ENTRY_ROOT="/opt/tiger/entry"
ENV_NAME="eval_harness"

# ------------------------------------------
# 1. 创建并激活 Python 3.10 环境（conda 优先，无 conda 则 venv）
# ------------------------------------------
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        conda create -n "${ENV_NAME}" python=3.10 -y
    fi
    conda activate "${ENV_NAME}"
else
    VENV_DIR="${ENTRY_ROOT}/${ENV_NAME}"
    if [[ ! -d "${VENV_DIR}" ]]; then
        python3.10 -m venv "${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"
fi

python --version
pip install --upgrade pip

# ------------------------------------------
# 2. 安装 lm-evaluation-harness（与原环境完全一致的步骤）
#       python 3.10
#       pip install -e .
#       pip install "lm_eval[hf]"
# ------------------------------------------
cd "${ENTRY_ROOT}/lm-evaluation-harness"

pip install -e .
pip install "lm_eval[hf]"

# ------------------------------------------
# 3. 安装包清单（pip install -e . + lm_eval[hf] 会自动拉取）
# ------------------------------------------
# torch              原环境 2.11.0 (CUDA 13)
# transformers       原环境 5.7.0
# tokenizers         原环境 0.22.2
# accelerate         原环境 1.13.0
# peft               原环境 0.19.1
# datasets           原环境 4.8.5
# huggingface_hub    原环境 1.13.0
# safetensors        原环境 0.7.0
# evaluate           原环境 0.4.6
# numpy / pandas / scikit-learn / scipy
# sympy / jinja2 / pyyaml / tqdm / regex / nltk
# rouge-score / sacrebleu / sqlitedict / zstandard / dill
# pytablewriter / word2number / more-itertools

# ------------------------------------------
# 4. 验证
# ------------------------------------------
python -c "import lm_eval, torch, transformers; \
    print('lm_eval', lm_eval.__version__); \
    print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); \
    print('transformers', transformers.__version__)"

lm_eval --help >/dev/null && echo "[OK] lm_eval CLI 可用"

echo "=================================================="
echo "环境安装完成。激活方式："
echo "  conda activate ${ENV_NAME}"
echo "  # 或"
echo "  source ${ENTRY_ROOT}/${ENV_NAME}/bin/activate"
echo "=================================================="
