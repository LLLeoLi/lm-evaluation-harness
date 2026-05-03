#!/bin/bash
# ==========================================
# 通用能力评测启动脚本（迁移到 /opt/tiger/entry 之后使用）
#
# 与 eval_results_general/run_self_distill_eval.sh 区分：
#   - 项目根：/opt/tiger/entry
#   - HF 缓存：/opt/tiger/entry/hf_cache
#   - 模型权重：/opt/tiger/entry/<model-dir>/checkpoint-25  (用户自己下载到 entry 下)
#   - 输出：/mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt/eval_results_general_<DATE>
#
# 用法：
#   conda activate eval_harness   # 或 source <venv>/bin/activate
#   bash /opt/tiger/entry/lm-evaluation-harness/run_self_distill_eval_tiger.sh
# ==========================================

# ------------------------------------------
# 路径与缓存
# ------------------------------------------
ENTRY_ROOT="/opt/tiger/entry"

export HF_HOME="${ENTRY_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${ENTRY_ROOT}/hf_cache/datasets"
# 数据集首次会从网络下载；如已离线缓存，可打开下面两行
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
export HF_ENDPOINT="https://hf-mirror.com"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# ------------------------------------------
# 输出目录（HDFS）
# ------------------------------------------
RUN_TAG="$(date +%m%d)"   # e.g. 0503
HDFS_ROOT="/mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt"
BASE_OUT_DIR="${HDFS_ROOT}/eval_results_general_${RUN_TAG}"
mkdir -p "${BASE_OUT_DIR}"

set +e

# ------------------------------------------
# 任务集合（与原脚本一致：通用能力，不跑 math）
# ------------------------------------------
TASKS="mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k"

# ------------------------------------------
# 通用评测函数（沿用原版）
# ------------------------------------------
run_eval() {
    local MODEL_PATH="$1"
    local GPU_ID="$2"

    MODEL_NAME=$(basename "$MODEL_PATH")
    if [[ "$MODEL_NAME" == checkpoint-* ]]; then
        MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
    fi
    SAFE_MODEL_NAME="${MODEL_NAME// /_}"

    if [[ "$MODEL_NAME" == *Qwen3* ]]; then
        CURRENT_MODEL_ARGS="pretrained=$MODEL_PATH,enable_thinking=False"
        echo "[GPU ${GPU_ID}] Qwen3 模型 ($MODEL_NAME)，使用 enable_thinking=False"
    else
        CURRENT_MODEL_ARGS="pretrained=$MODEL_PATH"
        echo "[GPU ${GPU_ID}] 模型 ($MODEL_NAME)"
    fi

    MODEL_OUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}"
    mkdir -p "$MODEL_OUT_DIR"

    LOG_DIR="$MODEL_OUT_DIR/logs"
    mkdir -p "$LOG_DIR"

    echo -e "\n===== $(date +%F_%T) $TASKS =====" >> "$LOG_DIR/summary_temp0.txt"
    echo -e "\n===== $(date +%F_%T) $TASKS =====" >> "$LOG_DIR/summary_temp1.txt"

    # Temperature = 0 (1 次)
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUT_FILE="$MODEL_OUT_DIR/results_temp0_${TIMESTAMP}.json"
    LOG_FILE="$LOG_DIR/eval_temp0_${TIMESTAMP}.log"

    echo "-> [GPU ${GPU_ID}] Temperature=0: $MODEL_NAME"
    CUDA_VISIBLE_DEVICES=${GPU_ID} lm_eval --model hf \
        --model_args "$CURRENT_MODEL_ARGS" \
        --tasks "$TASKS" \
        --device "cuda:0" \
        --batch_size 32 \
        --gen_kwargs temperature=0.0,do_sample=False \
        --output_path "$OUT_FILE" \
        2>&1 | tee "$LOG_FILE"

    echo -e "\n[GPU ${GPU_ID}] $MODEL_NAME (Temp=0) 结果:" >> "$LOG_DIR/summary_temp0.txt"
    grep -A 50 "| Task" "$LOG_FILE" >> "$LOG_DIR/summary_temp0.txt"

    # Temperature = 1 (3 次)
    for j in {1..3}; do
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        OUT_FILE="$MODEL_OUT_DIR/results_temp1_run${j}_${TIMESTAMP}.json"
        LOG_FILE="$LOG_DIR/eval_temp1_run${j}_${TIMESTAMP}.log"

        echo "-> [GPU ${GPU_ID}] Temperature=1 (第 $j/3 次): $MODEL_NAME"
        CUDA_VISIBLE_DEVICES=${GPU_ID} lm_eval --model hf \
            --model_args "$CURRENT_MODEL_ARGS" \
            --tasks "$TASKS" \
            --device "cuda:0" \
            --batch_size 32 \
            --gen_kwargs temperature=1.0,do_sample=True \
            --output_path "$OUT_FILE" \
            2>&1 | tee "$LOG_FILE"

        echo -e "\n[GPU ${GPU_ID}] $MODEL_NAME (Temp=1 Run $j) 结果:" >> "$LOG_DIR/summary_temp1.txt"
        grep -A 50 "| Task" "$LOG_FILE" >> "$LOG_DIR/summary_temp1.txt"
    done

    echo "[GPU ${GPU_ID}] 模型 $MODEL_NAME 评测完成！"
}

# ------------------------------------------
# 待评测模型列表
# 模型权重根目录由用户填好（下载到 ${ENTRY_ROOT} 下面）
# 每个 entry 指向具体的 checkpoint 目录
# ------------------------------------------
MODEL_ROOT="${ENTRY_ROOT}"   # TODO: 如有专门的 model_weight 子目录请改这里
CKPT="checkpoint-25"

ALL_MODELS=(
    # TODO: 填入实际下载好的模型路径，例：
    # "$MODEL_ROOT/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100/$CKPT"
    # "$MODEL_ROOT/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100/$CKPT"
)

# ------------------------------------------
# GPU 池与并行轮转
# ------------------------------------------
GPU_POOL=(0 1 2 3 4 5 6 7)   # TODO: 按实际可用 GPU 调整
NUM_GPUS=${#GPU_POOL[@]}
TOTAL=${#ALL_MODELS[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "[ERR] ALL_MODELS 为空，请填入模型路径后再运行"
    exit 1
fi

echo "=================================================="
echo "评测 $TOTAL 个模型 (GPU ${GPU_POOL[*]} 并行轮转)"
echo "输出：$BASE_OUT_DIR"
echo "=================================================="

for ((i=0; i<TOTAL; i+=NUM_GPUS)); do
    BATCH_END=$(( i + NUM_GPUS ))
    [[ $BATCH_END -gt $TOTAL ]] && BATCH_END=$TOTAL
    echo ">>> Batch: 模型索引 $i ~ $((BATCH_END-1))"
    for ((j=i; j<BATCH_END; j++)); do
        GPU_ID=${GPU_POOL[$(( j - i ))]}
        run_eval "${ALL_MODELS[$j]}" "$GPU_ID" &
    done
    wait
    echo ">>> Batch 完成: 模型索引 $i ~ $((BATCH_END-1))"
done

echo "=================================================="
echo "全部 $TOTAL 个模型评测完成！"
echo "=================================================="
