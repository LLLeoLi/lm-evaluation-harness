#!/bin/bash
# AlpacaEval 公用 helper, 由 run_0507_alpaca_<family>.sh source.
# 上层必须预先定义:
#   MODEL_BASE       e.g. "Llama-3-8B-Instruct"
#   MODEL_FAMILY     e.g. llama3_8b
#   GPU_POOL         e.g. (4 5 6 7)
#   HK_PAIRS         28 个 "h k" pair
# 可选:
#   MODEL_ROOT       默认 SDFT model_weight 目录
#   CKPT_SUBDIR      默认 checkpoint-25
#   BASE_OUT_DIR     默认 eval_results_alpaca_0507/<MODEL_FAMILY>
#   ALPACA_JSON      默认 hf_cache/alpaca_eval.json
#   JUDGE_CONFIG_DIR 默认 .../judge_configs/deepseek_chat
#   GEN_TEMP / GEN_TOP_P / GEN_MAX_TOKENS  generate 阶段采样参数

set -u
set +e

# ===== !!! 移植到新机器需修改 !!! =====
# 1) base_dir: 本仓库 (lm-evaluation-harness/) 所在的项目根目录 (含 hf_cache/ 等)
# 2) MODEL_ROOT: SDFT 训练好的权重目录 (driver 里也可覆盖)
# 3) VLLM_PY / ALPACA_EVAL_BIN (下方): vllm conda env 的 python / alpaca_eval CLI
#    本仓库 README 说明的 conda env: vllm (装了 vllm + alpaca-eval)
# ======================================
base_dir="/ssddata/lihao/projects/Self-Distillation-eval"
LME_DIR="${base_dir}/lm-evaluation-harness"
MODEL_ROOT="${MODEL_ROOT:-/ssddata/lihao/projects/SDFT-for-Safe-Alignment/model_weight}"
CKPT_SUBDIR="${CKPT_SUBDIR:-checkpoint-25}"
BASE_OUT_DIR="${BASE_OUT_DIR:-${LME_DIR}/eval_results_alpaca_0507/${MODEL_FAMILY}}"
ALPACA_JSON="${ALPACA_JSON:-${base_dir}/hf_cache/alpaca_eval.json}"
REFERENCE_JSON="${REFERENCE_JSON:-${base_dir}/hf_cache/alpaca_7b_baseline.json}"
JUDGE_CONFIG_DIR="${JUDGE_CONFIG_DIR:-${LME_DIR}/alpaca_eval/judge_configs/deepseek_chat}"
GEN_TEMP="${GEN_TEMP:-0.7}"
GEN_TOP_P="${GEN_TOP_P:-1.0}"
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-512}"
LIMIT="${LIMIT:-}"
LIMIT_ARGS=()
[[ -n "${LIMIT}" ]] && LIMIT_ARGS=( --limit "${LIMIT}" )

mkdir -p "${BASE_OUT_DIR}"

# conda env: vllm (含 vllm + alpaca-eval; 见 setup_0507.sh)
# 自动用 conda info --envs 解析路径; 不存在时报错
ENV_NAME_VLLM="${ENV_NAME_VLLM:-vllm}"
VLLM_ENV_DIR="$(conda info --envs 2>/dev/null | awk -v n="${ENV_NAME_VLLM}" '$1==n {print $NF}')"
if [ -z "${VLLM_ENV_DIR}" ]; then
    echo "[ERROR] 未找到 conda env '${ENV_NAME_VLLM}'; 请先跑 setup_0507.sh 或建好 env" >&2
    exit 1
fi
VLLM_PY="${VLLM_ENV_DIR}/bin/python"
ALPACA_EVAL_BIN="${VLLM_ENV_DIR}/bin/alpaca_eval"
GEN_SCRIPT="${LME_DIR}/alpaca_eval/generate_alpaca_0507.py"

# DeepSeek API (judge); 调用前必须 export OPENAI_API_KEY=<your_key>
# OPENAI_API_BASE 默认指向 DeepSeek 公开端点; 如需走其它兼容端点请覆盖
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.deepseek.com}"
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] 请先 export OPENAI_API_KEY=<deepseek key>" >&2
    exit 1
fi

# 模型列表
MODELS=()
for hk in "${HK_PAIRS[@]}"; do
    h="${hk% *}"; k="${hk#* }"
    MODELS+=( "${MODEL_ROOT}/${MODEL_BASE}-vector-mode2-top50-horizon${h}-alpha1.0-keep${k}-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100/${CKPT_SUBDIR}" )
done
TOTAL=${#MODELS[@]}
NUM_GPUS=${#GPU_POOL[@]}

# Phase 1: vLLM 生成 (一卡一模型, 4 卡轮转)
gen_one() {
    local MODEL_PATH="$1" GPU="$2"
    local NAME=$(basename "$(dirname "${MODEL_PATH}")")
    local OUT_DIR="${BASE_OUT_DIR}/${NAME}"
    local OUT_JSON="${OUT_DIR}/model_outputs.json"
    local LOG="${OUT_DIR}/gen.log"
    mkdir -p "${OUT_DIR}"
    if [ -f "${OUT_JSON}" ]; then
        echo "[skip-gen] ${NAME} (model_outputs.json 已存在)"
        return
    fi
    echo "-> [GPU ${GPU}] gen ${NAME}"
    "${VLLM_PY}" "${GEN_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --output_path "${OUT_JSON}" \
        --model_name "${NAME}" \
        --alpaca_json "${ALPACA_JSON}" \
        --device_ids "${GPU}" \
        --temperature "${GEN_TEMP}" \
        --top_p "${GEN_TOP_P}" \
        --max_tokens "${GEN_MAX_TOKENS}" \
        "${LIMIT_ARGS[@]}" \
        > "${LOG}" 2>&1
    echo "<- [GPU ${GPU}] gen ${NAME} done"
}

# Phase 2: alpaca_eval CLI (judge 不占 GPU, 顺序跑即可)
judge_one() {
    local MODEL_PATH="$1"
    local NAME=$(basename "$(dirname "${MODEL_PATH}")")
    local OUT_DIR="${BASE_OUT_DIR}/${NAME}"
    local OUT_JSON="${OUT_DIR}/model_outputs.json"
    local JUDGE_OUT="${OUT_DIR}/judge"
    local LOG="${OUT_DIR}/judge.log"
    if [ ! -f "${OUT_JSON}" ]; then
        echo "[skip-judge] ${NAME} (没有 model_outputs.json)"
        return
    fi
    if [ -f "${JUDGE_OUT}/leaderboard.csv" ]; then
        echo "[skip-judge] ${NAME} (leaderboard.csv 已存在)"
        return
    fi
    mkdir -p "${JUDGE_OUT}"
    echo "-> judge ${NAME}"
    "${ALPACA_EVAL_BIN}" \
        --model_outputs "${OUT_JSON}" \
        --reference_outputs "${REFERENCE_JSON}" \
        --annotators_config "${JUDGE_CONFIG_DIR}" \
        --output_path "${JUDGE_OUT}" \
        > "${LOG}" 2>&1
    echo "<- judge ${NAME} done"
}

echo "=================================================="
echo "[run_0507_alpaca] family=${MODEL_FAMILY} base=${MODEL_BASE}"
echo "  MODEL_ROOT      = ${MODEL_ROOT}"
echo "  GPU_POOL        = ${GPU_POOL[*]}  (${NUM_GPUS} 卡)"
echo "  TOTAL           = ${TOTAL} 模型"
echo "  ALPACA_JSON     = ${ALPACA_JSON}"
echo "  JUDGE_CONFIG    = ${JUDGE_CONFIG_DIR}"
echo "  GEN sampling    = T=${GEN_TEMP} top_p=${GEN_TOP_P} max=${GEN_MAX_TOKENS}"
echo "  OUT             = ${BASE_OUT_DIR}"
echo "=================================================="

# Phase 1: 4 卡轮转 vLLM 生成
echo ""
echo ">>> Phase 1: vLLM generate (${TOTAL} 模型 / ${NUM_GPUS} 卡)"
for ((i=0; i<TOTAL; i+=NUM_GPUS)); do
    BATCH_END=$(( i + NUM_GPUS ))
    [[ $BATCH_END -gt $TOTAL ]] && BATCH_END=$TOTAL
    for ((j=i; j<BATCH_END; j++)); do
        GPU_ID=${GPU_POOL[$(( j - i ))]}
        gen_one "${MODELS[$j]}" "$GPU_ID" &
    done
    wait
    echo ">>> Phase 1 batch 完成: 模型 $i ~ $((BATCH_END-1))"
done

# Phase 2: 顺序 judge (API 限速, 不并发以免被 ban)
echo ""
echo ">>> Phase 2: alpaca_eval judge (DeepSeek API, 顺序)"
for MODEL_PATH in "${MODELS[@]}"; do
    judge_one "${MODEL_PATH}"
done

echo "=================================================="
echo "[run_0507_alpaca] family=${MODEL_FAMILY} 完成. 结果在 ${BASE_OUT_DIR}/<model>/judge/"
echo "=================================================="
