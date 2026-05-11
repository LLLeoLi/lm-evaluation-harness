#!/bin/bash
# 集群侧通用能力评测 — 一次跑完 CKPT_GLOB 匹配的所有模型
# 用法:
#   bash run.sh                                     # 默认 Llama-3-8B-Instruct 系列
#   CKPT_BASE=Qwen3-8B bash run.sh                  # 切换基座
#   CKPT_GLOB="Llama-3-8B-Instruct-vector-mode2-*" bash run.sh
#   NUM_GPUS=4 LM_BATCH_SIZE=16 bash run.sh
#   LIMIT=20 bash run.sh                            # 烟雾测试 (每任务只跑前 20 题)
#
# 调用前需 `conda activate eval_harness`.
# 集群路径与 setup.sh 一致, 移植到本地需相应调整.

set +e
set -u

# ===== 路径 (集群侧, 与 setup.sh 一致) =====
ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"

export HF_HOME="${ENTRY_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# 减少显存碎片, 对变长 batch (loglikelihood) 收益明显
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# humaneval 用 evaluate.load("code_eval"), 它在 import 时就要求此 env var
# (lm_eval 的 --confirm_run_unsafe_code 是框架层, 跟这个不是一回事, 两者都要给)
export HF_ALLOW_CODE_EVAL=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# 清理半下载残留, 避免 datasets 抓 .incomplete 报错
find "${HF_DATASETS_CACHE}" -type d -name "*.incomplete" -exec rm -rf {} + 2>/dev/null

# ===== 模型来源 (HDFS) =====
CKPT_BASE="${CKPT_BASE:-Llama-3-8B-Instruct}"
CKPT_ROOT="/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts/${CKPT_BASE}"
CKPT_GLOB="${CKPT_GLOB:-${CKPT_BASE}-vector-*-PKU_UnSafeRLHF_100}"

HDFS_RUN_DIR="/mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt/eval_results_general_$(date +%m%d)"
mkdir -p "${HDFS_RUN_DIR}"
LOG_DIR="${LME_DIR}/logs/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# ===== 评测配置 =====
NUM_GPUS="${NUM_GPUS:-8}"
LM_BATCH_SIZE="${LM_BATCH_SIZE:-auto}"     # auto 让 lm_eval 自适应; 也可设 8/16/32
GEN_T1_REPEATS="${GEN_T1_REPEATS:-3}"      # generate_until 在 T=1 下的重复次数
LAUNCH_STAGGER="${LAUNCH_STAGGER:-5}"      # 启动间隔, 错开 ckpt shard 加载峰值

# humaneval 任务: instruct 模型用 humaneval_instruct, base 模型用 humaneval, "" 则不跑
HUMANEVAL_TASK="${HUMANEVAL_TASK-humaneval_instruct}"
LIMIT="${LIMIT:-}"
LIMIT_ARGS=()
[[ -n "${LIMIT}" ]] && LIMIT_ARGS=( --limit "${LIMIT}" )

# loglikelihood 任务集: temperature 无意义, 跑一次即可
LM_TASKS="mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2"
# generate_until 任务集: 受 temperature 影响, T=0 一次 + T=1 重复 GEN_T1_REPEATS 次
GEN_TASKS="gsm8k,hendrycks_math"
[[ -n "${HUMANEVAL_TASK}" ]] && GEN_TASKS="${GEN_TASKS},${HUMANEVAL_TASK}"

GEN_EXTRA_ARGS=()
[[ "${HUMANEVAL_TASK}" == humaneval* ]] && GEN_EXTRA_ARGS+=( --confirm_run_unsafe_code )

# ===== 模型发现 =====
shopt -s nullglob
MODELS=("${CKPT_ROOT}"/${CKPT_GLOB})
shopt -u nullglob

echo "=================================================="
echo "[run] CKPT_BASE       = ${CKPT_BASE}"
echo "[run] CKPT_ROOT       = ${CKPT_ROOT}"
echo "[run] CKPT_GLOB       = ${CKPT_GLOB}"
echo "[run] 发现模型数      = ${#MODELS[@]}"
echo "[run] NUM_GPUS        = ${NUM_GPUS}"
echo "[run] LM_BATCH_SIZE   = ${LM_BATCH_SIZE}"
echo "[run] LM_TASKS        = ${LM_TASKS}"
echo "[run] GEN_TASKS       = ${GEN_TASKS}"
echo "[run] GEN_T1_REPEATS  = ${GEN_T1_REPEATS}"
echo "[run] HUMANEVAL_TASK  = ${HUMANEVAL_TASK:-<不跑>}"
echo "[run] HDFS_RUN_DIR    = ${HDFS_RUN_DIR}"
echo "[run] LOG_DIR         = ${LOG_DIR}"
[[ -n "${LIMIT}" ]] && echo "[run] LIMIT           = ${LIMIT} (烟雾测试)"
echo "=================================================="

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "[run] WARN: 没有匹配 ${CKPT_GLOB} 的模型, 退出"
    exit 0
fi

cd "${LME_DIR}"

# ===== 单次 lm_eval 调用 =====
run_lm() {
    # loglikelihood 任务集, 不传 gen_kwargs
    local ARGS=$1 OUT_DIR=$2 NAME=$3 GPU=$4
    local TS=$(date +"%Y%m%d_%H%M%S")
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${LM_TASKS}" \
        --device cuda:0 --batch_size "${LM_BATCH_SIZE}" \
        "${LIMIT_ARGS[@]}" \
        --output_path "${OUT_DIR}/results_lm_${TS}.json" \
        >> "${LOG_DIR}/${NAME}_lm_${TS}.log" 2>&1
}

run_gen() {
    # generate_until 任务集, 单一温度配置
    local TEMP=$1 SUFFIX=$2 ARGS=$3 OUT_DIR=$4 NAME=$5 GPU=$6
    local TS=$(date +"%Y%m%d_%H%M%S")
    local GEN_KWARGS
    if [ "${TEMP}" = "0.0" ]; then
        GEN_KWARGS="do_sample=False"
    else
        GEN_KWARGS="temperature=${TEMP},do_sample=True"
    fi
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${GEN_TASKS}" \
        --device cuda:0 --batch_size "${LM_BATCH_SIZE}" \
        --gen_kwargs "${GEN_KWARGS}" \
        "${GEN_EXTRA_ARGS[@]}" \
        "${LIMIT_ARGS[@]}" \
        --output_path "${OUT_DIR}/results_${SUFFIX}_${TS}.json" \
        >> "${LOG_DIR}/${NAME}_${SUFFIX}_${TS}.log" 2>&1
}

run_model_on_gpu() {
    local EVAL_PATH=$1 GPU=$2 NAME=$3 ARGS=$4 OUT_DIR=$5
    echo ">>> [GPU ${GPU}] ${NAME} → $(basename ${EVAL_PATH})"
    run_lm  "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
    run_gen 0.0 gen_temp0 "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
    for ((j=1; j<=GEN_T1_REPEATS; j++)); do
        run_gen 1.0 "gen_temp1_run${j}" "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
    done
    echo "<<< [GPU ${GPU}] ${NAME} 完成"
}

# ===== GPU 池 (空闲即取, 比固定 batch 利用率高) =====
declare -A GPU_PID
for ((g=0; g<NUM_GPUS; g++)); do GPU_PID[$g]=0; done
ACQUIRED_GPU=-1

# 通过全局变量 ACQUIRED_GPU 返回, 避免 $(acquire_gpu) 进子 shell 后 wait -n 看不到父 shell 的后台进程
acquire_gpu() {
    while true; do
        for ((g=0; g<NUM_GPUS; g++)); do
            local pid=${GPU_PID[$g]}
            if [ "$pid" = "0" ] || ! kill -0 "$pid" 2>/dev/null; then
                GPU_PID[$g]=0
                ACQUIRED_GPU=$g
                return
            fi
        done
        wait -n 2>/dev/null || sleep 2
    done
}

# ===== 主循环 =====
for MODEL_PATH in "${MODELS[@]}"; do
    NAME=$(basename "${MODEL_PATH}")

    # 选 step 数最大的 checkpoint-*
    EVAL_PATH=""; LATEST=-1
    shopt -s nullglob
    for c in "${MODEL_PATH}"/checkpoint-*; do
        [ -d "$c" ] || continue
        step="${c##*/checkpoint-}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        (( step > LATEST )) && LATEST=$step && EVAL_PATH="$c"
    done
    shopt -u nullglob
    if [ -z "${EVAL_PATH}" ]; then
        echo "[run] WARN: ${NAME} 无 checkpoint, 跳过"
        continue
    fi

    ARGS="pretrained=${EVAL_PATH}"
    [[ "${NAME}" == *Qwen3* ]] && ARGS="${ARGS},enable_thinking=False"
    OUT_DIR="${HDFS_RUN_DIR}/${NAME}"
    mkdir -p "${OUT_DIR}"

    acquire_gpu
    GPU_ID=$ACQUIRED_GPU
    run_model_on_gpu "${EVAL_PATH}" "${GPU_ID}" "${NAME}" "${ARGS}" "${OUT_DIR}" &
    GPU_PID[$GPU_ID]=$!
    sleep "${LAUNCH_STAGGER}"
done

wait
echo "=================================================="
echo "[run] 全部完成! 结果 → ${HDFS_RUN_DIR}"
echo "[run] 日志        → ${LOG_DIR}"
echo "=================================================="
