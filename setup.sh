#!/bin/bash
# 一次性环境/数据准备 — 集群侧 (/opt/tiger/entry/...)
# 用法: bash setup.sh
# 重复执行安全: 每一步先检测, 已存在则跳过.
#
# 前置:
#   - 已建好 conda env `eval_harness` (lm-evaluation-harness 主 env, 装 lm_eval CLI).
#   - 已建好 conda env `vllm` (装了 vllm). 本脚本会再向其装 alpaca-eval.
#
# ===== !!! 移植到新机器需修改 !!! =====
# 1) ENTRY_DIR: 集群入口目录 (含 lm-evaluation-harness/, hf_cache/)
# 2) CLUSTER_DATA_SCRIPT: 集群侧批量下数据集脚本 (没有就置空)
# 3) ENV_NAME_EVAL / ENV_NAME_VLLM: 如两个 env 名称不同请修改
# ======================================

set -u

# ===== 路径基准 (集群侧, 与 run.sh 一致) =====
ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"
HF_HOME_DIR="${ENTRY_DIR}/hf_cache"
HUB_DIR="${HF_HOME_DIR}/hub"
CLUSTER_DATA_SCRIPT="${ENTRY_DIR}/scripts/download_dataset.sh"
ENV_NAME_EVAL="eval_harness"
ENV_NAME_VLLM="vllm"

mkdir -p "${HF_HOME_DIR}" "${HUB_DIR}"

# ===== conda 自检 + 激活 eval_harness =====
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到 conda; 请先安装 miniconda/anaconda" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME_EVAL} "; then
    echo "[ERROR] conda env '${ENV_NAME_EVAL}' 不存在; 请先创建" >&2
    exit 1
fi
conda activate "${ENV_NAME_EVAL}"

# vllm env 路径 (用 conda info 解析, 不写死)
VLLM_ENV_DIR="$(conda info --envs | awk -v n="${ENV_NAME_VLLM}" '$1==n {print $NF}')"
if [ -z "${VLLM_ENV_DIR}" ]; then
    echo "[WARN] conda env '${ENV_NAME_VLLM}' 不存在; 跳过 alpaca-eval 安装"
    vllm_pip=""
else
    vllm_pip="${VLLM_ENV_DIR}/bin/pip"
fi

# ===== HF 缓存设置 (下载阶段需要联网) =====
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME="${HF_HOME_DIR}"
export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"

echo "================================================="
echo "[setup] ENTRY_DIR  = ${ENTRY_DIR}"
echo "[setup] LME_DIR    = ${LME_DIR}"
echo "[setup] HF_HOME    = ${HF_HOME}"
echo "[setup] eval env   = ${ENV_NAME_EVAL} ($(which python))"
echo "[setup] vllm env   = ${ENV_NAME_VLLM} (${VLLM_ENV_DIR:-<missing>})"
echo "================================================="

# ---------- 1) 安装 lm-evaluation-harness ----------
if python -c "import lm_eval" >/dev/null 2>&1; then
    echo "[skip] lm_eval 已安装"
else
    echo "[install] lm-evaluation-harness (editable) ..."
    ( cd "${LME_DIR}" && pip install -e . --user ) \
        || { echo "[ERROR] lm-evaluation-harness 安装失败"; exit 1; }
    pip install "lm_eval[hf]" --user \
        || { echo "[ERROR] lm_eval[hf] 安装失败"; exit 1; }
fi

# ---------- 2) 集群侧批量下数据集 (如果存在) ----------
if [ -x "${CLUSTER_DATA_SCRIPT}" ] || [ -f "${CLUSTER_DATA_SCRIPT}" ]; then
    cluster_marker="${HF_HOME_DIR}/.cluster_data_done"
    if [ -f "${cluster_marker}" ]; then
        echo "[skip] 集群 download_dataset.sh 已运行过"
    else
        echo "[run] ${CLUSTER_DATA_SCRIPT} ..."
        bash "${CLUSTER_DATA_SCRIPT}" \
            && touch "${cluster_marker}" \
            || { echo "[ERROR] ${CLUSTER_DATA_SCRIPT} 执行失败"; exit 1; }
    fi
else
    echo "[skip] ${CLUSTER_DATA_SCRIPT} 不存在; 走标准 HF 下载"
fi

# ---------- 3) HF 数据集预下载 (load_dataset 触发 builder cache) ----------
DATASETS=(
    "cais/mmlu"
    "openai/gsm8k"
    "EleutherAI/hendrycks_math"
    "openai/openai_humaneval"
)
for repo in "${DATASETS[@]}"; do
    builder_name="${repo//\//___}"
    if [ -d "${HF_DATASETS_CACHE}/${builder_name}" ]; then
        echo "[skip] dataset ${repo} 已有 builder cache"
    else
        echo "[download] dataset ${repo} ..."
        python -c "from datasets import load_dataset; load_dataset('${repo}'); print('[ok] ${repo}')" \
            || { echo "[ERROR] 下载/build ${repo} 失败"; exit 1; }
    fi
done

# ---------- 4) evaluate metric: code_eval (HumanEval pass@k 用) ----------
code_eval_marker="${HF_HOME_DIR}/.code_eval_cached"
if [ -f "${code_eval_marker}" ]; then
    echo "[skip] evaluate metric code_eval 已 cache"
else
    echo "[download] evaluate metric code_eval ..."
    python -c "import evaluate; evaluate.load('code_eval'); print('[ok] code_eval loaded')" \
        && touch "${code_eval_marker}" \
        || { echo "[ERROR] code_eval 下载失败"; exit 1; }
fi

# ---------- 5) AlpacaEval (装到 vllm env, generate + judge 共用) ----------
if [ -n "${vllm_pip}" ]; then
    if "${vllm_pip}" show alpaca-eval >/dev/null 2>&1; then
        echo "[skip] alpaca-eval 已装入 ${ENV_NAME_VLLM} env"
    else
        echo "[install] alpaca-eval → ${ENV_NAME_VLLM} env"
        "${vllm_pip}" install alpaca-eval \
            || { echo "[ERROR] alpaca-eval 安装失败"; exit 1; }
    fi

    for fname in alpaca_eval.json alpaca_eval_gpt4_baseline.json; do
        fpath="${HF_HOME_DIR}/${fname}"
        if [ -f "${fpath}" ]; then
            echo "[skip] ${fname} 已 cache"
        else
            echo "[download] ${fname} ..."
            python -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id='tatsu-lab/alpaca_eval', filename='${fname}', repo_type='dataset')
shutil.copy(p, '${fpath}')
print('[ok] ${fname} →', '${fpath}')
" || { echo "[ERROR] ${fname} 下载失败"; exit 1; }
        fi
    done

    # 修补 alpaca_eval judge config 里的绝对路径 (yaml 不支持 env 变量插值)
    JUDGE_CFG="${LME_DIR}/alpaca_eval/judge_configs/deepseek_chat/configs.yaml"
    PROMPT_PATH="${LME_DIR}/alpaca_eval/judge_configs/deepseek_chat/alpaca_eval.txt"
    if [ -f "${JUDGE_CFG}" ]; then
        if grep -qF "prompt_template: \"${PROMPT_PATH}\"" "${JUDGE_CFG}"; then
            echo "[skip] judge config prompt_template 已正确"
        else
            echo "[fix] 改写 judge config prompt_template → ${PROMPT_PATH}"
            sed -i "s|^  prompt_template: \".*\"|  prompt_template: \"${PROMPT_PATH}\"|" "${JUDGE_CFG}"
        fi
    fi
fi

echo ""
echo "================================================="
echo "[setup] 完成. 跑评测前需要的 env (run.sh 已自动设):"
echo "  export HF_HOME=${HF_HOME}"
echo "  export HF_DATASETS_CACHE=\${HF_HOME}/datasets"
echo "  export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  export HF_ALLOW_CODE_EVAL=1   # humaneval 必需"
echo "================================================="
