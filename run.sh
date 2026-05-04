#!/bin/bash
# ==============================================================================
# run.sh — lm-evaluation-harness 通用能力评测
#   每个训练产物自动选 step 号最大的 checkpoint, 跑 lm_eval (temp=0 一次 + temp=1 三次)
# ==============================================================================
set +e

ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"

export HF_HOME="${ENTRY_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

CKPT_ROOT="/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts"
CKPT_GLOB="${CKPT_GLOB:-Llama-3-8B-Instruct-vector-*-PKU_UnSafeRLHF_100}"

HDFS_RUN_DIR="/mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt/eval_results_general_$(date +%m%d)"
LOG_DIR="${HDFS_RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

BATCH_SIZE=8
TASKS="mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k"

shopt -s nullglob
MODELS=("${CKPT_ROOT}"/${CKPT_GLOB})
shopt -u nullglob
echo "[run] 发现 ${#MODELS[@]} 个模型, 输出 → ${HDFS_RUN_DIR}"

cd "${LME_DIR}"

run_one() {
    local TEMP=$1 SUFFIX=$2 ARGS=$3 OUT_DIR=$4 GPU=$5
    local TS=$(date +"%Y%m%d_%H%M%S")
    local DO_SAMPLE=False; [ "${TEMP}" = "1.0" ] && DO_SAMPLE=True
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${TASKS}" \
        --device cuda:0 --batch_size 32 \
        --gen_kwargs "temperature=${TEMP},do_sample=${DO_SAMPLE}" \
        --output_path "${OUT_DIR}/results_${SUFFIX}_${TS}.json" \
        > "${OUT_DIR}/logs/eval_${SUFFIX}_${TS}.log" 2>&1
}

for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    GPU_ID=$(( i % BATCH_SIZE ))
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
    [ -z "${EVAL_PATH}" ] && { echo "[run] WARN: ${NAME} 无 checkpoint, 跳过"; continue; }

    ARGS="pretrained=${EVAL_PATH}"
    [[ "${NAME}" == *Qwen3* ]] && ARGS="${ARGS},enable_thinking=False"
    OUT_DIR="${HDFS_RUN_DIR}/${NAME}"
    mkdir -p "${OUT_DIR}/logs"

    echo ">>> [GPU ${GPU_ID}] ${NAME} → $(basename ${EVAL_PATH})"
    (
        run_one 0.0 temp0      "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
        run_one 1.0 temp1_run1 "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
        run_one 1.0 temp1_run2 "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
        run_one 1.0 temp1_run3 "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
    ) &

    (( (i + 1) % BATCH_SIZE == 0 )) && wait
done
wait

echo "[run] 全部完成! ${HDFS_RUN_DIR}"
