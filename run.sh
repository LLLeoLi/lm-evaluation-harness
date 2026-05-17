#!/bin/bash
# ==============================================================================
# run.sh — 集群侧通用能力评测; 自动从 ${CKPT_ROOT} 下按 ${CKPT_GLOB} 检索模型,
#         每个模型再自动取 step 数最大的 checkpoint-*.
#
#   与 run_models.sh 流程一致, 唯一区别是模型来源 (run_models.sh 直接接受路径列表).
#
#   任务集 (0507 版本, 与 _run_0507_common.sh 对齐):
#     LM  (loglikelihood, 跑 1 次):           默认空 (置 LM_TASKS=mmlu 开启)
#     GEN (generate_until, T=0 + T=1×N):     默认 hendrycks_math + humaneval_instruct
#
#   用法:
#     bash run.sh                                     # 默认 Llama-3-8B-Instruct 系列
#     CKPT_BASE=Qwen3-8B bash run.sh                  # 切换基座
#     CKPT_GLOB="Llama-3-8B-Instruct-vector-mode2-*" bash run.sh
#     NUM_GPUS=4 LM_BATCH_SIZE=16 bash run.sh
#     LIMIT=20 bash run.sh                            # 烟雾测试 (每任务只跑前 20 题)
#
#   可选环境变量:
#     LM_TASKS         默认空字符串 (不跑 loglikelihood); 设为 mmlu 等可启用
#     GEN_TASKS        默认 hendrycks_math (会自动追加 HUMANEVAL_TASK)
#     HUMANEVAL_TASK   默认 humaneval_instruct (base 模型改用 humaneval; 设为 "" 不跑)
#     GEN_T1_REPEATS   GEN_TASKS 在 T=1 下重复次数, 默认 0 (即只跑 T=0)
#     LM_BATCH_SIZE    lm_eval --batch_size, 默认 auto
#     LIMIT            每任务样本上限 (烟雾测试)
#
#   输出:
#     ${HDFS_RUN_DIR}/<NAME>/${EVAL_SUBDIR}/{lm,gen_temp0,gen_temp1_run1,...}.json
#     ${HDFS_RUN_DIR}/eval_summary_${DATE_TAG}.{csv,json}
# ==============================================================================
set +e
set -u

# ===== 路径 (集群侧, 与 setup.sh 一致) =====
ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"

# 可用 GPU 数: 优先 NUM_GPUS env; 否则按 CUDA_VISIBLE_DEVICES 推断;
# 仍未定再 fallback 到 nvidia-smi 统计的物理卡数; 都拿不到才默认 8.
if [ -z "${NUM_GPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a _cvd_arr <<< "${CUDA_VISIBLE_DEVICES}"
        NUM_GPUS=${#_cvd_arr[@]}
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
        [ "${NUM_GPUS}" -eq 0 ] && NUM_GPUS=8
    else
        NUM_GPUS=8
    fi
fi
echo "[run] NUM_GPUS=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"

export HF_HOME="${ENTRY_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# 0507 版本: 完整离线三件套 (与 _run_0507_common.sh 对齐)
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 减少显存碎片, 对变长 batch (loglikelihood) 收益明显
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
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

DATE_TAG=$(date +%Y%m%d)
EVAL_SUBDIR="eval_general_${DATE_TAG}"
HDFS_RUN_DIR="${HDFS_RUN_DIR:-/mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt/eval_results_general_0507_${DATE_TAG}}"
mkdir -p "${HDFS_RUN_DIR}"
LOG_DIR="${LME_DIR}/logs/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# ===== 评测配置 (0507 默认) =====
LM_BATCH_SIZE="${LM_BATCH_SIZE:-auto}"     # auto 让 lm_eval 自适应; 也可设 8/16/32
LAUNCH_STAGGER="${LAUNCH_STAGGER:-5}"      # 启动间隔, 错开 ckpt shard 加载峰值

# loglikelihood (mmlu): 当前默认跳过 (置空); 如需打开置 LM_TASKS=mmlu
LM_TASKS="${LM_TASKS:-}"
# generate_until: 当前默认只跑 T=0 (GEN_T1_REPEATS=0); 默认任务集去掉 gsm8k
# humaneval: instruct 模型用 humaneval_instruct (默认), base 用 humaneval; 空字符串则不跑
HUMANEVAL_TASK="${HUMANEVAL_TASK-humaneval_instruct}"
GEN_TASKS="${GEN_TASKS:-hendrycks_math}"
[[ -n "${HUMANEVAL_TASK}" ]] && GEN_TASKS="${GEN_TASKS},${HUMANEVAL_TASK}"
GEN_T1_REPEATS="${GEN_T1_REPEATS:-0}"

LIMIT="${LIMIT:-}"
LIMIT_ARGS=()
[[ -n "${LIMIT}" ]] && LIMIT_ARGS=( --limit "${LIMIT}" )

# humaneval (任一变体) 需要 --confirm_run_unsafe_code
GEN_EXTRA_ARGS=()
[[ "${GEN_TASKS}" == *humaneval* ]] && GEN_EXTRA_ARGS+=( --confirm_run_unsafe_code )

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
echo "[run] LM_TASKS        = ${LM_TASKS:-<不跑>}"
echo "[run] GEN_TASKS       = ${GEN_TASKS:-<不跑>}"
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
# loglikelihood 任务集 (mmlu 等), 不传 gen_kwargs
run_lm() {
    local ARGS=$1 OUT_DIR=$2 NAME=$3 GPU=$4
    local TS=$(date +"%Y%m%d_%H%M%S")
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${LM_TASKS}" \
        --device cuda:0 --batch_size "${LM_BATCH_SIZE}" \
        "${LIMIT_ARGS[@]}" \
        --output_path "${OUT_DIR}/results_lm_${TS}.json" \
        >> "${LOG_DIR}/${NAME}_lm_${TS}.log" 2>&1
}

# generate_until 任务集, 单一温度配置
run_gen() {
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
    [[ -n "${LM_TASKS}" ]]  && run_lm  "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
    [[ -n "${GEN_TASKS}" ]] && run_gen 0.0 gen_temp0 "${ARGS}" "${OUT_DIR}" "${NAME}" "${GPU}"
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
EVAL_BASES=()
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
    OUT_BASE="${HDFS_RUN_DIR}/${NAME}/${EVAL_SUBDIR}"
    mkdir -p "${OUT_BASE}"
    EVAL_BASES+=("${OUT_BASE}")

    acquire_gpu
    GPU_ID=$ACQUIRED_GPU
    run_model_on_gpu "${EVAL_PATH}" "${GPU_ID}" "${NAME}" "${ARGS}" "${OUT_BASE}" &
    GPU_PID[$GPU_ID]=$!
    sleep "${LAUNCH_STAGGER}"
done

wait
echo "[run] 推理阶段完成"

# ============ 汇总: 把所有模型每个 task 的主指标拢到一张表 ============
SUMMARY_CSV="${HDFS_RUN_DIR}/eval_summary_${DATE_TAG}.csv"
SUMMARY_JSON="${HDFS_RUN_DIR}/eval_summary_${DATE_TAG}.json"

PARENTS=()
for OUT_BASE in "${EVAL_BASES[@]}"; do
    PARENTS+=("$(dirname "${OUT_BASE}")")
done

python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${EVAL_SUBDIR}" "${PARENTS[@]}" <<'PY'
import csv, json, os, sys

csv_path, json_path, eval_subdir, *model_paths = sys.argv[1:]

# 选主指标的优先级 (lm-eval 不同 task 字段名不一样)
PREFERRED = [
    "acc,none", "acc_norm,none",
    "exact_match,strict-match", "exact_match,flexible-extract",
    "mc2,none", "em,none",
]

def pick_primary(metrics):
    for k in PREFERRED:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            return k, v
    for k, v in metrics.items():
        if k.endswith("_stderr,none") or "stderr" in k:
            continue
        if isinstance(v, (int, float)):
            return k, v
    return None, None

rows = []
for mp in model_paths:
    name = os.path.basename(mp.rstrip("/"))
    eval_dir = os.path.join(mp, eval_subdir)
    if not os.path.isdir(eval_dir):
        continue
    for root, _, files in os.walk(eval_dir):
        if os.path.basename(root) == "logs":
            continue
        for fn in files:
            if not fn.endswith(".json") or "samples" in fn:
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[summary] WARN: 读取失败 {fp}: {e}")
                continue
            if not isinstance(data, dict):
                continue
            results = data.get("results")
            if not isinstance(results, dict):
                continue
            rel = os.path.relpath(fp, eval_dir)
            top = rel.split(os.sep)[0]
            if top.endswith(".json"):
                split = top[:-5].removeprefix("results_")
            else:
                split = top.removeprefix("results_")
            for task, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                metric_key, value = pick_primary(metrics)
                if value is None:
                    continue
                rows.append({
                    "model": name,
                    "split": split,
                    "task": task,
                    "metric": metric_key,
                    "value": float(value),
                })

rows.sort(key=lambda r: (r["model"], r["split"], r["task"]))

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "split", "task", "metric", "value"])
    for r in rows:
        w.writerow([r["model"], r["split"], r["task"], r["metric"], f"{r['value']:.4f}"])

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("\n========== Eval Summary ==========")
print(f"{'model':<55} {'split':<14} {'task':<25} {'metric':<30} {'value':>8}")
for r in rows:
    print(f"{r['model']:<55} {r['split']:<14} {r['task']:<25} {r['metric']:<30} {r['value']:>8.4f}")
print(f"\n[summary] CSV  → {csv_path}")
print(f"[summary] JSON → {json_path}")
PY

echo "=================================================="
echo "[run] 全部完成! 结果 → ${HDFS_RUN_DIR}/<NAME>/${EVAL_SUBDIR}/"
echo "[run] 日志        → ${LOG_DIR}"
echo "=================================================="
