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
export HF_DATASETS_OFFLINE=1   # 完全靠本地 cache; 先跑 scripts/download_dataset.sh 把数据备齐
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# 清理上次中断遗留的 *.incomplete 目录
find "${HF_DATASETS_CACHE}" -type d -name "*.incomplete" -exec rm -rf {} + 2>/dev/null

# 训练脚本把产物放到 sf_ckpts/${MODEL_BASE}/${MODEL_BASE}-vector-...
# 切换被测 family: CKPT_BASE=Qwen3-8B bash run.sh
CKPT_BASE="${CKPT_BASE:-Llama-3-8B-Instruct}"
CKPT_ROOT="/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts/${CKPT_BASE}"
CKPT_GLOB="${CKPT_GLOB:-${CKPT_BASE}-vector-*-PKU_UnSafeRLHF_100}"

DATE_TAG=$(date +%Y%m%d)
EVAL_SUBDIR="eval_general_${DATE_TAG}"

BATCH_SIZE=8
TASKS="mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k"

shopt -s nullglob
MODELS=("${CKPT_ROOT}"/${CKPT_GLOB})
shopt -u nullglob
echo "[run] 发现 ${#MODELS[@]} 个模型, 输出 → 各模型目录下 ${EVAL_SUBDIR}/"

cd "${LME_DIR}"

run_one() {
    local TEMP=$1 SUFFIX=$2 ARGS=$3 OUT_DIR=$4 GPU=$5
    local DO_SAMPLE=False; [ "${TEMP}" = "1.0" ] && DO_SAMPLE=True
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${TASKS}" \
        --device cuda:0 --batch_size 32 \
        --gen_kwargs "temperature=${TEMP},do_sample=${DO_SAMPLE}" \
        --output_path "${OUT_DIR}/results_${SUFFIX}.json" \
        > "${OUT_DIR}/logs/${SUFFIX}.log" 2>&1
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
    OUT_DIR="${MODEL_PATH}/${EVAL_SUBDIR}"
    mkdir -p "${OUT_DIR}/logs"

    echo ">>> [GPU ${GPU_ID}] ${NAME} → $(basename ${EVAL_PATH})  (log: ${OUT_DIR}/logs/*.log)"
    (
        run_one 0.0 temp0      "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
        run_one 1.0 temp1_run1 "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
        run_one 1.0 temp1_run2 "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
        run_one 1.0 temp1_run3 "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
    ) &

    (( (i + 1) % BATCH_SIZE == 0 )) && wait
done
wait

echo "[run] 全部完成! 结果在各模型目录下 ${EVAL_SUBDIR}/"
