#!/bin/bash
# 公用 helper, 由 run_0507_<family>.sh source.
# 上层必须预先定义:
#   MODEL_BASE       例如 "Llama-3-8B-Instruct" / "Qwen3-8B" / "Llama-3.2-3B-Instruct" / "Qwen3-4B"
#   MODEL_FAMILY     例如 llama3_8b / qwen3_8b / llama3_3b / qwen3_4b   (用于输出目录)
#   GPU_POOL         例如 (4 5 6 7)
#   HK_PAIRS         28 个 "h k" pair (LARF run_safe_system_prompt_new.sh 同款)
# 可选:
#   MODEL_ROOT       默认 /ssddata/lihao/projects/SDFT-for-Safe-Alignment/model_weight
#   CKPT_SUBDIR      默认 checkpoint-25
#   BATCH_SIZE       lm_eval --batch_size, 默认 32
#   BASE_OUT_DIR     默认 eval_results_0507/<MODEL_FAMILY>
#   GEN_T1_REPEATS   GEN_TASKS 在 T=1 下的重复次数, 默认 3
#   HUMANEVAL_TASK   humaneval 任务名: instruct 模型用 humaneval_instruct (默认),
#                    base 模型用 humaneval; 设为 "" 则不跑 humaneval
#   LIMIT            每个任务只跑前 N 道题 (调试/烟雾测试用); 默认空 = 全量

set -u
set +e

# ===== !!! 移植到新机器需修改 !!! =====
# 1) base_dir: 本仓库 (lm-evaluation-harness/) 所在的项目根目录 (含 hf_cache/)
# 2) MODEL_ROOT (driver 中, 默认 /ssddata/.../SDFT-for-Safe-Alignment/model_weight)
# 3) 调用前必须 `conda activate eval_harness` (env 名见 README; lm_eval CLI 装在此 env)
# ======================================
# ===== 缓存与离线 (与 run.sh 一致) =====
base_dir="/ssddata/lihao/projects/Self-Distillation-eval"
export HF_HOME="${base_dir}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# humaneval 用 evaluate 库的 code_eval metric, 它在 import 时就要求此 env var
# (lm_eval 的 --confirm_run_unsafe_code 是框架层, 跟这个不是一回事, 两者都要给)
export HF_ALLOW_CODE_EVAL=1

MODEL_ROOT="${MODEL_ROOT:-/ssddata/lihao/projects/SDFT-for-Safe-Alignment/model_weight}"
CKPT_SUBDIR="${CKPT_SUBDIR:-checkpoint-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BASE_OUT_DIR="${BASE_OUT_DIR:-${base_dir}/lm-evaluation-harness/eval_results_0507/${MODEL_FAMILY}}"
GEN_T1_REPEATS="${GEN_T1_REPEATS:-3}"
HUMANEVAL_TASK="${HUMANEVAL_TASK-humaneval_instruct}"
LIMIT="${LIMIT:-}"
LIMIT_ARGS=()
[[ -n "${LIMIT}" ]] && LIMIT_ARGS=( --limit "${LIMIT}" )
mkdir -p "${BASE_OUT_DIR}"

# ===== 任务划分 (run.sh 风格) =====
# loglikelihood: temperature 无意义, 跑 1 次
LM_TASKS="mmlu"
# generate_until: 受 temperature 影响, T=0 一次 + T=1 重复 GEN_T1_REPEATS 次
GEN_TASKS="gsm8k,hendrycks_math"
[[ -n "${HUMANEVAL_TASK}" ]] && GEN_TASKS="${GEN_TASKS},${HUMANEVAL_TASK}"

# humaneval (任一变体) 需要 --confirm_run_unsafe_code
GEN_EXTRA_ARGS=()
[[ "${HUMANEVAL_TASK}" == humaneval* ]] && GEN_EXTRA_ARGS+=( --confirm_run_unsafe_code )

# ===== 模型列表 =====
MODELS=()
for hk in "${HK_PAIRS[@]}"; do
    h="${hk% *}"; k="${hk#* }"
    MODELS+=( "${MODEL_ROOT}/${MODEL_BASE}-vector-mode2-top50-horizon${h}-alpha1.0-keep${k}-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100/${CKPT_SUBDIR}" )
done
TOTAL=${#MODELS[@]}
NUM_GPUS=${#GPU_POOL[@]}

# ===== 单次 lm_eval 调用 =====
# loglikelihood 任务集, 不传 gen_kwargs
run_lm() {
    local ARGS=$1 OUT_DIR=$2 LOG_DIR=$3 NAME=$4 GPU=$5
    local TS=$(date +"%Y%m%d_%H%M%S")
    echo "-> [GPU ${GPU}] ${NAME}  LM (${LM_TASKS})"
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${LM_TASKS}" \
        --device cuda:0 --batch_size "${BATCH_SIZE}" \
        "${LIMIT_ARGS[@]}" \
        --output_path "${OUT_DIR}/results_lm_${TS}.json" \
        2>&1 | tee "${LOG_DIR}/lm_${TS}.log" >/dev/null
}

# generate_until 任务集, 单一温度配置
run_gen() {
    local TEMP=$1 SUFFIX=$2 ARGS=$3 OUT_DIR=$4 LOG_DIR=$5 NAME=$6 GPU=$7
    local TS=$(date +"%Y%m%d_%H%M%S")
    local GEN_KWARGS
    if [ "${TEMP}" = "0.0" ]; then
        GEN_KWARGS="do_sample=False"
    else
        GEN_KWARGS="temperature=${TEMP},do_sample=True"
    fi
    echo "-> [GPU ${GPU}] ${NAME}  GEN ${SUFFIX} (T=${TEMP})"
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${GEN_TASKS}" \
        --device cuda:0 --batch_size "${BATCH_SIZE}" \
        --gen_kwargs "${GEN_KWARGS}" \
        "${GEN_EXTRA_ARGS[@]}" \
        "${LIMIT_ARGS[@]}" \
        --output_path "${OUT_DIR}/results_${SUFFIX}_${TS}.json" \
        2>&1 | tee "${LOG_DIR}/${SUFFIX}_${TS}.log" >/dev/null
}

# ===== 单模型完整流程 =====
run_eval() {
    local MODEL_PATH="$1" GPU="$2"
    local NAME=$(basename "$(dirname "${MODEL_PATH}")")
    local SAFE_NAME="${NAME// /_}"
    local OUT_DIR="${BASE_OUT_DIR}/${SAFE_NAME}"
    local LOG_DIR="${OUT_DIR}/logs"
    mkdir -p "${LOG_DIR}"

    local ARGS="pretrained=${MODEL_PATH}"

    echo ">>> [GPU ${GPU}] ${NAME}"
    run_lm  "${ARGS}" "${OUT_DIR}" "${LOG_DIR}" "${NAME}" "${GPU}"
    run_gen 0.0 gen_temp0 "${ARGS}" "${OUT_DIR}" "${LOG_DIR}" "${NAME}" "${GPU}"
    for ((j=1; j<=GEN_T1_REPEATS; j++)); do
        run_gen 1.0 "gen_temp1_run${j}" "${ARGS}" "${OUT_DIR}" "${LOG_DIR}" "${NAME}" "${GPU}"
    done
    # AlpacaEval (NSPO) — TODO: 单独阶段, 需 vLLM serve + script/alpaca_eval.sh
    echo "<<< [GPU ${GPU}] ${NAME} 完成"
}

# ===== 4 卡轮转主循环 =====
echo "=================================================="
echo "[run_0507] family=${MODEL_FAMILY} base=${MODEL_BASE}"
echo "  MODEL_ROOT      = ${MODEL_ROOT}"
echo "  GPU_POOL        = ${GPU_POOL[*]}  (${NUM_GPUS} 卡)"
echo "  TOTAL           = ${TOTAL} 模型"
echo "  LM_TASKS        = ${LM_TASKS}"
echo "  GEN_TASKS       = ${GEN_TASKS}"
echo "  GEN_T1_REPEATS  = ${GEN_T1_REPEATS}"
echo "  HUMANEVAL_TASK  = ${HUMANEVAL_TASK:-<不跑>}"
echo "  OUT             = ${BASE_OUT_DIR}"
echo "=================================================="

for ((i=0; i<TOTAL; i+=NUM_GPUS)); do
    BATCH_END=$(( i + NUM_GPUS ))
    [[ $BATCH_END -gt $TOTAL ]] && BATCH_END=$TOTAL
    echo ">>> Batch: 模型索引 $i ~ $((BATCH_END-1))"
    for ((j=i; j<BATCH_END; j++)); do
        GPU_ID=${GPU_POOL[$(( j - i ))]}
        run_eval "${MODELS[$j]}" "$GPU_ID" &
    done
    wait
    echo ">>> Batch 完成: 模型索引 $i ~ $((BATCH_END-1))"
done

echo "=================================================="
echo "[run_0507] family=${MODEL_FAMILY} 全部 $TOTAL 个模型完成"
echo "=================================================="
