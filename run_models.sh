#!/bin/bash
# ==============================================================================
# run_models.sh — 对一组(已存在的) HF 模型直接做 lm-eval-harness 通用能力评测
#   与 run.sh 流程一致, 但模型来源不同:
#     run.sh:        从 ${CKPT_ROOT} 下按 ${CKPT_GLOB} 检索, 并自动取最新 checkpoint-*
#     run_models.sh: 通过命令行参数 / MODELS 环境变量 / -f 文件 直接给定模型路径
#                    (路径直接作为 lm_eval 的 pretrained=, 不再向下找 checkpoint-*)
#
#   任务集 (0507 版本, 与 _run_0507_common.sh 对齐):
#     LM (loglikelihood, 跑 1 次):           mmlu
#     GEN (generate_until, T=0 + T=1×N):     gsm8k, hendrycks_math, humaneval_instruct
#
#   用法:
#     bash run_models.sh <model_path_or_id> [<model_path_or_id> ...]
#     MODELS="m1 m2 m3" bash run_models.sh
#     bash run_models.sh -f models.txt           # 从文件读, 每行一个模型
#
#   可选环境变量:
#     LM_TASKS         默认 mmlu
#     GEN_TASKS        默认 gsm8k,hendrycks_math (会自动追加 HUMANEVAL_TASK)
#     HUMANEVAL_TASK   默认 humaneval_instruct (base 模型改用 humaneval; 设为 "" 则不跑)
#     GEN_T1_REPEATS   GEN_TASKS 在 T=1 下重复次数, 默认 3
#     BATCH_SIZE_LM    lm_eval --batch_size, 默认 32
#
#   输出:
#     ${OUTPUT_ROOT}/<basename(model)>/${EVAL_SUBDIR}/{lm,gen_temp0,gen_temp1_run1,...}.json
#     ${OUTPUT_ROOT}/eval_summary_${DATE_TAG}.{csv,json}
# ==============================================================================
set +e
set -u

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
echo "[run_models] NUM_GPUS=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"

export HF_HOME="${ENTRY_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# 0507 版本: 完整离线三件套 (与 _run_0507_common.sh 对齐)
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# humaneval 用 evaluate 库的 code_eval metric, import 时就要求此 env var
# (--confirm_run_unsafe_code 是框架层, 两者都要给)
export HF_ALLOW_CODE_EVAL=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# 清理上次中断遗留的 *.incomplete 目录
find "${HF_DATASETS_CACHE}" -type d -name "*.incomplete" -exec rm -rf {} + 2>/dev/null

DATE_TAG=$(date +%Y%m%d)
EVAL_SUBDIR="eval_general_${DATE_TAG}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts/_models_eval_general_0507_${DATE_TAG}}"

# ===== 0507 任务划分 =====
# loglikelihood (mmlu): 当前默认跳过 (置空); 如需打开置 LM_TASKS=mmlu
LM_TASKS="${LM_TASKS:-}"
# generate_until: 当前默认只跑 T=0 (GEN_T1_REPEATS=0); 默认任务集去掉 gsm8k
# humaneval: instruct 模型用 humaneval_instruct (默认), base 用 humaneval; 空字符串则不跑
HUMANEVAL_TASK="${HUMANEVAL_TASK-humaneval_instruct}"
GEN_TASKS="${GEN_TASKS:-hendrycks_math}"
[[ -n "${HUMANEVAL_TASK}" ]] && GEN_TASKS="${GEN_TASKS},${HUMANEVAL_TASK}"
GEN_T1_REPEATS="${GEN_T1_REPEATS:-0}"
BATCH_SIZE_LM="${BATCH_SIZE_LM:-32}"

# humaneval (任一变体) 需要 --confirm_run_unsafe_code
GEN_EXTRA_ARGS=()
[[ "${GEN_TASKS}" == *humaneval* ]] && GEN_EXTRA_ARGS+=( --confirm_run_unsafe_code )

# ------------- 解析模型列表 -------------
MODELS_INPUT=()
if [ "${1:-}" = "-f" ] && [ -n "${2:-}" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="${line//$'\r'/}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -n "$line" ] && MODELS_INPUT+=("$line")
    done < "$2"
elif [ "$#" -gt 0 ]; then
    MODELS_INPUT=("$@")
elif [ -n "${MODELS_ENV:-${MODELS:-}}" ]; then
    # 兼容: MODELS="m1 m2" bash run_models.sh
    read -r -a MODELS_INPUT <<< "${MODELS_ENV:-${MODELS}}"
fi

if [ "${#MODELS_INPUT[@]}" -eq 0 ]; then
    echo "[run_models] 用法: bash $0 <model_path> [<model_path> ...]"
    echo "             或: MODELS=\"m1 m2\" bash $0"
    echo "             或: bash $0 -f models.txt"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[run_models] 共 ${#MODELS_INPUT[@]} 个模型, 输出根目录: ${OUTPUT_ROOT}"
for m in "${MODELS_INPUT[@]}"; do echo "  - $m"; done

cd "${LME_DIR}"

# loglikelihood 任务集 (mmlu), 不传 gen_kwargs
run_lm() {
    local ARGS=$1 OUT_DIR=$2 GPU=$3
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${LM_TASKS}" \
        --device cuda:0 --batch_size "${BATCH_SIZE_LM}" \
        --output_path "${OUT_DIR}/results_lm.json" \
        > "${OUT_DIR}/logs/lm.log" 2>&1
}

# generate_until 任务集 (gsm8k, hendrycks_math, humaneval_instruct), 单一温度配置
run_gen() {
    local TEMP=$1 SUFFIX=$2 ARGS=$3 OUT_DIR=$4 GPU=$5
    local GEN_KWARGS
    if [ "${TEMP}" = "0.0" ]; then
        GEN_KWARGS="do_sample=False"
    else
        GEN_KWARGS="temperature=${TEMP},do_sample=True"
    fi
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${GEN_TASKS}" \
        --device cuda:0 --batch_size "${BATCH_SIZE_LM}" \
        --gen_kwargs "${GEN_KWARGS}" \
        "${GEN_EXTRA_ARGS[@]}" \
        --output_path "${OUT_DIR}/results_${SUFFIX}.json" \
        > "${OUT_DIR}/logs/${SUFFIX}.log" 2>&1
}

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

LAUNCH_STAGGER="${LAUNCH_STAGGER:-5}"  # 启动间隔, 错开 ckpt shard 加载峰值

# 记录 (model_path, out_base) 对供阶段 2 汇总复用
EVAL_BASES=()
EVAL_NAMES=()

declare -A SEEN_NAMES
for MODEL_PATH in "${MODELS_INPUT[@]}"; do
    # 去掉末尾 / 后取 basename, 作为输出目录名
    _MP="${MODEL_PATH%/}"
    NAME=$(basename "${_MP}")
    # 若 basename 形如 checkpoint-XXX, 拼上上一级目录名以避免不同模型同名冲突
    if [[ "${NAME}" == checkpoint-* ]]; then
        PARENT=$(basename "$(dirname "${_MP}")")
        NAME="${PARENT}-${NAME}"
    fi
    # 通用去重: 若该 NAME 已被占用 (不同 MODEL_PATH 同 basename, 例如多个 .../<algo>/ppo-lag),
    # 不断向上拼父目录, 直到唯一; 路径根都用尽则附加序号
    if [ -n "${SEEN_NAMES[${NAME}]:-}" ] && [ "${SEEN_NAMES[${NAME}]}" != "${MODEL_PATH}" ]; then
        _D="$(dirname "${_MP}")"
        while [ -n "${SEEN_NAMES[${NAME}]:-}" ] && [ "${SEEN_NAMES[${NAME}]}" != "${MODEL_PATH}" ] && [ "${_D}" != "/" ] && [ "${_D}" != "." ]; do
            NAME="$(basename "${_D}")-${NAME}"
            _D="$(dirname "${_D}")"
        done
        _i=2
        while [ -n "${SEEN_NAMES[${NAME}]:-}" ] && [ "${SEEN_NAMES[${NAME}]}" != "${MODEL_PATH}" ]; do
            NAME="${NAME}_${_i}"
            _i=$((_i+1))
        done
    fi
    SEEN_NAMES[${NAME}]="${MODEL_PATH}"

    if [ ! -e "${MODEL_PATH}" ]; then
        # 不存在的本地路径仍允许 (可能是 HF id), 仅打印提示
        echo "[run_models] NOTE: ${MODEL_PATH} 不是本地路径, 当作 HF model id 处理"
    fi

    ARGS="pretrained=${MODEL_PATH}"
    [[ "${NAME}" == *Qwen3* ]] && ARGS="${ARGS},enable_thinking=False"

    OUT_BASE="${OUTPUT_ROOT}/${NAME}/${EVAL_SUBDIR}"
    mkdir -p "${OUT_BASE}/logs"
    EVAL_BASES+=("${OUT_BASE}")
    EVAL_NAMES+=("${NAME}")

    acquire_gpu
    GPU_ID=$ACQUIRED_GPU
    echo ">>> [GPU ${GPU_ID}] ${NAME} → ${MODEL_PATH}  (log: ${OUT_BASE}/logs/*.log)"
    (
        [[ -n "${LM_TASKS}" ]] && run_lm "${ARGS}" "${OUT_BASE}" "${GPU_ID}"
        [[ -n "${GEN_TASKS}" ]] && run_gen 0.0 gen_temp0 "${ARGS}" "${OUT_BASE}" "${GPU_ID}"
        for ((j=1; j<=GEN_T1_REPEATS; j++)); do
            run_gen 1.0 "gen_temp1_run${j}" "${ARGS}" "${OUT_BASE}" "${GPU_ID}"
        done
    ) &
    GPU_PID[${GPU_ID}]=$!
    sleep "${LAUNCH_STAGGER}"
done

wait
echo "[run_models] 推理阶段完成"

# ============ 汇总: 把所有模型每个 task 的主指标拢到一张表 ============
SUMMARY_CSV="${OUTPUT_ROOT}/eval_summary_${DATE_TAG}.csv"
SUMMARY_JSON="${OUTPUT_ROOT}/eval_summary_${DATE_TAG}.json"

# 把每个模型的 "<OUTPUT_ROOT>/<NAME>" 父目录传给 python, 复用 run.sh 的汇总逻辑
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

echo "[run_models] 全部完成! 结果在 ${OUTPUT_ROOT}/<model>/${EVAL_SUBDIR}/"
