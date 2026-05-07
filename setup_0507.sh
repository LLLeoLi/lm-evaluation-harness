#!/bin/bash
# 0507 通用能力评测 — 一次性环境/数据准备
# 用法: bash setup_0507.sh
# 重复执行安全: 每一步先检测, 已存在则跳过.
#
# 前置:
#   - 已建好 conda env `eval_harness` (跑 setup.sh 即可),
#     里面装了 lm-evaluation-harness + datasets + evaluate.
#   - 已建好 conda env `vllm` (装了 vllm). setup_0507.sh 会再向其装 alpaca-eval.
#
# ===== !!! 移植到新机器需修改 !!! =====
# 1) base_dir: 本仓库所在项目根目录 (会在下方建 hf_cache/)
# 2) ENV_NAME_EVAL / ENV_NAME_VLLM: 如果你两个 env 不叫这名字
# (其它 path: HF_HOME 自动派生自 base_dir; conda 路径 conda info --base 自动找)
# ======================================

set -u

# ===== 路径基准 =====
base_dir="/ssddata/lihao/projects/Self-Distillation-eval"
hf_home="${base_dir}/hf_cache"
hub_dir="${hf_home}/hub"
ENV_NAME_EVAL="eval_harness"
ENV_NAME_VLLM="vllm"

mkdir -p "${hf_home}" "${hub_dir}"

# ===== conda 自检 + 激活 eval_harness =====
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到 conda; 请先安装 miniconda/anaconda" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME_EVAL} "; then
    echo "[ERROR] conda env '${ENV_NAME_EVAL}' 不存在; 请先跑 setup.sh 创建" >&2
    exit 1
fi
conda activate "${ENV_NAME_EVAL}"

# vllm env 路径 (用 conda info 解析, 不写死)
VLLM_ENV_DIR="$(conda info --envs | awk -v n="${ENV_NAME_VLLM}" '$1==n {print $NF}')"
if [ -z "${VLLM_ENV_DIR}" ]; then
    echo "[ERROR] conda env '${ENV_NAME_VLLM}' 不存在; 请先建 (装 vllm)" >&2
    exit 1
fi
vllm_pip="${VLLM_ENV_DIR}/bin/pip"
vllm_python="${VLLM_ENV_DIR}/bin/python"

# ===== HF 缓存设置 =====
# 下载阶段需要联网, 显式 unset OFFLINE 三件套
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME="${hf_home}"
export HF_DATASETS_CACHE="${hf_home}/datasets"

echo "================================================="
echo "[setup_0507] base_dir   = ${base_dir}"
echo "[setup_0507] HF_HOME    = ${HF_HOME}"
echo "[setup_0507] eval env   = ${ENV_NAME_EVAL} ($(which python))"
echo "[setup_0507] vllm env   = ${ENV_NAME_VLLM} (${VLLM_ENV_DIR})"
echo "================================================="

# ---------- 1) HF 数据集预下载 ----------
# lm-eval task → HF repo:
#   mmlu / gsm8k / hendrycks_math / humaneval(openai/openai_humaneval)
DATASETS=(
    "cais/mmlu"
    "openai/gsm8k"
    "EleutherAI/hendrycks_math"
    "openai/openai_humaneval"
)

for repo in "${DATASETS[@]}"; do
    # 检查 datasets builder cache (hf_cache/datasets/<owner>___<name>/)
    # —— hub/ 下只是 raw parquet, 不够; lm_eval 用 datasets.load_dataset 必须有 builder cache
    builder_name="${repo//\//___}"
    if [ -d "${HF_DATASETS_CACHE}/${builder_name}" ]; then
        echo "[skip] dataset ${repo} 已有 builder cache"
    else
        echo "[download] dataset ${repo} (load_dataset 触发 build) ..."
        python -c "from datasets import load_dataset; load_dataset('${repo}'); print('[ok] ${repo}')" \
            || { echo "[ERROR] 下载/build ${repo} 失败"; exit 1; }
    fi
done

# ---------- 2) evaluate metric: code_eval (HumanEval pass@k 用) ----------
# evaluate.load("code_eval") 第一次会去 HF Spaces 拉 metric 脚本; 离线模式会失败.
# 因为已 export HF_HOME, 实际落盘到 ${HF_HOME}/modules/evaluate_modules/
#   metrics/evaluate-metric--code_eval/<hash>/code_eval.py
code_eval_marker="${HF_HOME}/.code_eval_cached"
if [ -f "${code_eval_marker}" ]; then
    echo "[skip] evaluate metric code_eval 已 cache"
else
    echo "[download] evaluate metric code_eval ..."
    python -c "import evaluate; evaluate.load('code_eval'); print('[ok] code_eval loaded')" \
        && touch "${code_eval_marker}" \
        || { echo "[ERROR] code_eval 下载失败"; exit 1; }
fi

# ---------- 3) AlpacaEval ----------
# generate 阶段用 vllm env, judge 阶段用 alpaca-eval CLI (装到 vllm env 即可共用)
if "${vllm_pip}" show alpaca-eval >/dev/null 2>&1; then
    echo "[skip] alpaca-eval 已装入 ${ENV_NAME_VLLM} env"
else
    echo "[install] alpaca-eval → ${ENV_NAME_VLLM} env"
    "${vllm_pip}" install alpaca-eval \
        || { echo "[ERROR] alpaca-eval 安装失败"; exit 1; }
fi

# tatsu-lab/alpaca_eval 数据集 + reference outputs
# 注意: 这是脚本式 dataset, datasets>=3.x 已 drop 支持. 直接拉 raw json 用.
for fname in alpaca_eval.json alpaca_eval_gpt4_baseline.json; do
    fpath="${HF_HOME}/${fname}"
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

# ---------- 4) 修补 alpaca_eval judge config 里的绝对路径 ----------
# alpaca_eval CLI 不支持 yaml 里的 env 变量插值, 只能用绝对路径.
# 这里根据 base_dir 自动改写 prompt_template 行, 避免移植时手改.
JUDGE_CFG="${base_dir}/lm-evaluation-harness/alpaca_eval/judge_configs/deepseek_chat/configs.yaml"
PROMPT_PATH="${base_dir}/lm-evaluation-harness/alpaca_eval/judge_configs/deepseek_chat/alpaca_eval.txt"
if [ -f "${JUDGE_CFG}" ]; then
    if grep -qF "prompt_template: \"${PROMPT_PATH}\"" "${JUDGE_CFG}"; then
        echo "[skip] judge config prompt_template 已正确"
    else
        echo "[fix] 改写 judge config prompt_template → ${PROMPT_PATH}"
        sed -i "s|^  prompt_template: \".*\"|  prompt_template: \"${PROMPT_PATH}\"|" "${JUDGE_CFG}"
    fi
fi

echo ""
echo "================================================="
echo "[setup_0507] 完成. 后续跑评测前请 export (run_0507_*.sh 已自动设):"
echo "  export HF_HOME=${HF_HOME}"
echo "  export HF_DATASETS_CACHE=\${HF_HOME}/datasets"
echo "  export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "================================================="
