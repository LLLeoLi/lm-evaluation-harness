set +e
set -u

ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"

export HF_HOME="${ENTRY_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# 减少显存碎片, 对变长 batch (loglikelihood) 收益明显
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

find "${HF_DATASETS_CACHE}" -type d -name "*.incomplete" -exec rm -rf {} + 2>/dev/null

CKPT_BASE="${CKPT_BASE:-Llama-3-8B-Instruct}"
CKPT_ROOT="/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts/${CKPT_BASE}"
CKPT_GLOB="${CKPT_GLOB:-${CKPT_BASE}-vector-*-PKU_UnSafeRLHF_100}"

HDFS_RUN_DIR="/mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt/eval_results_general_$(date +%m%d)"
mkdir -p "${HDFS_RUN_DIR}"
LOG_DIR="${LME_DIR}/logs/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "[run] 日志 → ${LOG_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"
LM_BATCH_SIZE="${LM_BATCH_SIZE:-auto}"   # auto 让 lm_eval 自适应; 也可设 8/16

# loglikelihood 任务: temperature 无意义, 跑一次即可
LM_TASKS="mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2"
# generate_until 任务: 受 temperature 影响, 需要重复采样
GEN_TASKS="gsm8k"

shopt -s nullglob
MODELS=("${CKPT_ROOT}"/${CKPT_GLOB})
shopt -u nullglob
echo "[run] 发现 ${#MODELS[@]} 个模型, 输出 → ${HDFS_RUN_DIR}"

cd "${LME_DIR}"

run_lm() {
    # 跑 loglikelihood 任务集, 不传 gen_kwargs
    local ARGS=$1 OUT_DIR=$2 NAME=$3 GPU=$4
    local TS=$(date +"%Y%m%d_%H%M%S")
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${LM_TASKS}" \
        --device cuda:0 --batch_size "${LM_BATCH_SIZE}" \
        --output_path "${OUT_DIR}/results_lm_${TS}.json" \
        >> "${LOG_DIR}/${NAME}_lm_${TS}.log" 2>&1
}

run_gen() {
    # 跑 generate_until 任务 (gsm8k), 单一温度配置
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
        --output_path "${OUT_DIR}/results_${SUFFIX}_${TS}.json" \
        >> "${LOG_DIR}/${NAME}_${SUFFIX}_${TS}.log" 2>&1
}

run_model_on_gpu() {
    local EVAL_PATH=$1 GPU=$2 NAME=$3 ARGS=$4 OUT_DIR=$5
    echo ">>> [GPU ${GPU}] ${NAME} → $(basename ${EVAL_PATH})"
    run_lm  "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
    run_gen 0.0 gsm8k_temp0      "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
    run_gen 1.0 gsm8k_temp1_run1 "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
    run_gen 1.0 gsm8k_temp1_run2 "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
    run_gen 1.0 gsm8k_temp1_run3 "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
    echo "<<< [GPU ${GPU}] ${NAME} 完成"
}

declare -A GPU_PID
for ((g=0; g<NUM_GPUS; g++)); do GPU_PID[$g]=0; done

acquire_gpu() {
    while true; do
        for ((g=0; g<NUM_GPUS; g++)); do
            local pid=${GPU_PID[$g]}
            if [ "$pid" = "0" ] || ! kill -0 "$pid" 2>/dev/null; then
                GPU_PID[$g]=0
                echo $g
                return
            fi
        done
        wait -n 2>/dev/null || sleep 2
    done
}

LAUNCH_STAGGER="${LAUNCH_STAGGER:-5}"  # 启动间隔, 错开 ckpt shard 加载峰值

for MODEL_PATH in "${MODELS[@]}"; do
    NAME=$(basename "${MODEL_PATH}")

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

    GPU_ID=$(acquire_gpu)
    run_model_on_gpu "${EVAL_PATH}" "${GPU_ID}" "${NAME}" "${ARGS}" "${OUT_DIR}" &
    GPU_PID[$GPU_ID]=$!
    sleep "${LAUNCH_STAGGER}"
done

wait
echo "[run] 全部完成! ${HDFS_RUN_DIR}"
