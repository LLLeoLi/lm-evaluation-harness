#!/bin/bash
# 一次性环境/数据准备 — 集群侧 (/opt/tiger/entry/...)
# 用法: bash setup.sh
# 重复执行安全: 每一步先检测, 已存在则跳过.
#
# 集群环境已就绪, 不需要 conda; 本脚本直接用当前 python/pip.
#
# ===== !!! 移植到新机器需修改 !!! =====
# 1) ENTRY_DIR: 集群入口目录 (含 lm-evaluation-harness/, hf_cache/)
# 2) CLUSTER_DATA_SCRIPT: 集群侧批量下数据集脚本 (没有就置空)
# ======================================

set -u

# ===== 路径基准 (集群侧, 与 run.sh 一致) =====
ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"
HF_HOME_DIR="${ENTRY_DIR}/hf_cache"
HUB_DIR="${HF_HOME_DIR}/hub"
CLUSTER_DATA_SCRIPT="${ENTRY_DIR}/scripts/download_dataset.sh"

mkdir -p "${HF_HOME_DIR}" "${HUB_DIR}"

# ===== python/pip 自检 (使用集群环境) =====
if ! command -v python >/dev/null 2>&1; then
    echo "[ERROR] 未找到 python" >&2
    exit 1
fi
if ! command -v pip >/dev/null 2>&1; then
    echo "[ERROR] 未找到 pip" >&2
    exit 1
fi
PIP="pip"

# ===== HF 缓存设置 (下载阶段需要联网) =====
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME="${HF_HOME_DIR}"
export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"

echo "================================================="
echo "[setup] ENTRY_DIR  = ${ENTRY_DIR}"
echo "[setup] LME_DIR    = ${LME_DIR}"
echo "[setup] HF_HOME    = ${HF_HOME}"
echo "[setup] python     = $(which python)"
echo "[setup] pip        = $(which pip)"
echo "================================================="

# ---------- 1) 安装 lm-evaluation-harness ----------
if python -c "import lm_eval" >/dev/null 2>&1; then
    echo "[skip] lm_eval 已安装"
else
    echo "[install] lm-evaluation-harness (editable) ..."
    ( cd "${LME_DIR}" && ${PIP} install -e . --user ) \
        || { echo "[ERROR] lm-evaluation-harness 安装失败"; exit 1; }
    ${PIP} install "lm_eval[hf]" --user \
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
# 形如 "repo" 或 "repo:cfg1,cfg2,..."  (后者表示该 dataset 必须按 config 加载)
DATASETS=(
    "cais/mmlu"
    "openai/gsm8k"
    "EleutherAI/hendrycks_math:algebra,counting_and_probability,geometry,intermediate_algebra,number_theory,prealgebra,precalculus"
    "openai/openai_humaneval"
)
for entry in "${DATASETS[@]}"; do
    repo="${entry%%:*}"
    cfgs="${entry#*:}"
    [ "${cfgs}" = "${entry}" ] && cfgs=""
    builder_name="${repo//\//___}"
    if [ -d "${HF_DATASETS_CACHE}/${builder_name}" ]; then
        echo "[skip] dataset ${repo} 已有 builder cache"
        continue
    fi
    if [ -z "${cfgs}" ]; then
        echo "[download] dataset ${repo} ..."
        python -c "from datasets import load_dataset; load_dataset('${repo}'); print('[ok] ${repo}')" \
            || { echo "[ERROR] 下载/build ${repo} 失败"; exit 1; }
    else
        IFS=',' read -r -a cfg_arr <<< "${cfgs}"
        for cfg in "${cfg_arr[@]}"; do
            echo "[download] dataset ${repo} (config=${cfg}) ..."
            python -c "from datasets import load_dataset; load_dataset('${repo}', '${cfg}'); print('[ok] ${repo}/${cfg}')" \
                || { echo "[ERROR] 下载/build ${repo}/${cfg} 失败"; exit 1; }
        done
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

# ---------- 5) AlpacaEval (generate + judge 共用) ----------
if ${PIP} show alpaca-eval >/dev/null 2>&1; then
    echo "[skip] alpaca-eval 已安装"
else
    echo "[install] alpaca-eval ..."
    ${PIP} install alpaca-eval --user \
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

# Alpaca-7B 参考输出 (win-rate vs Alpaca-7B 的 baseline). tatsu-lab/alpaca_eval HF
# dataset 里没有这份文件, 直接从 GitHub raw 拉.
alpaca_7b_path="${HF_HOME_DIR}/alpaca_7b_baseline.json"
if [ -f "${alpaca_7b_path}" ]; then
    echo "[skip] alpaca_7b_baseline.json 已 cache"
else
    echo "[download] alpaca_7b_baseline.json (Alpaca-7B reference outputs) ..."
    curl -fsSL \
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/alpaca-7b/model_outputs.json" \
        -o "${alpaca_7b_path}" \
        || { echo "[ERROR] alpaca_7b_baseline.json 下载失败"; exit 1; }
    echo "[ok] alpaca_7b_baseline.json → ${alpaca_7b_path}"
fi

# length-controlled winrate 计算时 alpaca_eval/metrics/glm_winrate.py 会
# hf_hub_download 这个校准文件; 离线模式下会炸, 这里提前预热进 HF cache.
df_gamed_marker="${HF_HOME_DIR}/.df_gamed_cached"
if [ -f "${df_gamed_marker}" ]; then
    echo "[skip] df_gamed.csv 已 cache"
else
    echo "[download] df_gamed.csv (alpaca_eval length-controlled winrate 校准文件) ..."
    python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='tatsu-lab/alpaca_eval', filename='df_gamed.csv', repo_type='dataset')
print('[ok] df_gamed.csv →', p)
" && touch "${df_gamed_marker}" \
    || { echo "[ERROR] df_gamed.csv 下载失败"; exit 1; }
fi

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

echo ""
echo "================================================="
echo "[setup] 完成. 跑评测前需要的 env (run.sh 已自动设):"
echo "  export HF_HOME=${HF_HOME}"
echo "  export HF_DATASETS_CACHE=\${HF_HOME}/datasets"
echo "  export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  export HF_ALLOW_CODE_EVAL=1   # humaneval 必需"
echo "================================================="
