#!/bin/bash
# ==============================================================================
# run_models_math500.sh — run_models.sh 的 math500-only 精简版
#   只跑 lm-eval-harness 的 minerva_math500 (不走 fewshot_as_multiturn)
#   不跑 AlpacaEval / mmlu / gsm8k / humaneval
#
#   默认: T=0 一次 + T=1 GEN_T1_REPEATS 次 (默认 1), 应用 chat template.
#   设 NO_CHAT_TEMPLATE=1 时则不传 --apply_chat_template (用于和 base 模板对比).
#
#   用法: 与 run_models.sh 完全一致
#     bash run_models_math500.sh <model_path> [<model_path> ...]
#     MODELS="m1 m2"  bash run_models_math500.sh
#     bash run_models_math500.sh -f models.txt
#
#   常用环境变量:
#     GEN_T1_REPEATS     T=1 重复次数, 默认 1 (即 T=0 + T=1 各 1 次)
#     BATCH_SIZE_LM      lm_eval --batch_size, 默认 32
#     NO_CHAT_TEMPLATE   =1 时跳过 --apply_chat_template
#     OUTPUT_ROOT        输出根目录
#
#   输出:
#     ${OUTPUT_ROOT}/<basename(model)>/${EVAL_SUBDIR}/results_{gen_temp0_nomt,gen_temp1_run<j>_nomt}.json
#     ${OUTPUT_ROOT}/eval_summary_math500_${DATE_TAG}.{csv,json}
# ==============================================================================
set +e
set -u

ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"

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
echo "[run_models_math500] NUM_GPUS=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"

export HF_HOME="${ENTRY_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

find "${HF_DATASETS_CACHE}" -type d -name "*.incomplete" -exec rm -rf {} + 2>/dev/null

DATE_TAG=20250516
EVAL_SUBDIR="eval_math500_${DATE_TAG}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts/_models_eval_math500_${DATE_TAG}}"

GEN_TASK="minerva_math500"
GEN_T1_REPEATS="${GEN_T1_REPEATS:-1}"
BATCH_SIZE_LM="${BATCH_SIZE_LM:-32}"
NO_CHAT_TEMPLATE="${NO_CHAT_TEMPLATE:-0}"

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
    read -r -a MODELS_INPUT <<< "${MODELS_ENV:-${MODELS}}"
fi

if [ "${#MODELS_INPUT[@]}" -eq 0 ]; then
    echo "[run_models_math500] 用法: bash $0 <model_path> [<model_path> ...]"
    echo "                     或: MODELS=\"m1 m2\" bash $0"
    echo "                     或: bash $0 -f models.txt"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[run_models_math500] 共 ${#MODELS_INPUT[@]} 个模型, 输出根目录: ${OUTPUT_ROOT}"
for m in "${MODELS_INPUT[@]}"; do echo "  - $m"; done

cd "${LME_DIR}"

# minerva_math500: generate_until, 不走 --fewshot_as_multiturn
run_gen_math500() {
    local TEMP=$1 SUFFIX=$2 ARGS=$3 OUT_DIR=$4 GPU=$5
    local GEN_KWARGS
    if [ "${TEMP}" = "0.0" ]; then
        GEN_KWARGS="do_sample=False"
    else
        GEN_KWARGS="temperature=${TEMP},do_sample=True"
    fi
    local EXTRA_ARGS=()
    [ "${NO_CHAT_TEMPLATE}" = "0" ] && EXTRA_ARGS+=( --apply_chat_template )
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${GEN_TASK}" \
        --device cuda:0 --batch_size "${BATCH_SIZE_LM}" \
        --gen_kwargs "${GEN_KWARGS}" \
        "${EXTRA_ARGS[@]}" \
        --output_path "${OUT_DIR}/results_${SUFFIX}.json" \
        > "${OUT_DIR}/logs/${SUFFIX}.log" 2>&1
}

declare -A GPU_PID
for ((g=0; g<NUM_GPUS; g++)); do GPU_PID[$g]=0; done
ACQUIRED_GPU=-1

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

LAUNCH_STAGGER="${LAUNCH_STAGGER:-5}"

# ============================================================================
# 阶段 0: 预先解析每个模型的 NAME / OUT_BASE
# ============================================================================
EVAL_BASES=()
EVAL_NAMES=()
declare -A SEEN_NAMES
for MODEL_PATH in "${MODELS_INPUT[@]}"; do
    _MP="${MODEL_PATH%/}"
    NAME=$(basename "${_MP}")
    if [[ "${NAME}" == checkpoint-* ]]; then
        PARENT=$(basename "$(dirname "${_MP}")")
        NAME="${PARENT}-${NAME}"
    fi
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
        echo "[run_models_math500] NOTE: ${MODEL_PATH} 不是本地路径, 当作 HF model id 处理"
    fi

    OUT_BASE="${OUTPUT_ROOT}/${NAME}/${EVAL_SUBDIR}"
    EVAL_BASES+=("${OUT_BASE}")
    EVAL_NAMES+=("${NAME}")
done

# ============================================================================
# 阶段 1: minerva_math500
# ============================================================================
echo ""
echo ">>> math500 Phase: ${#EVAL_NAMES[@]} 模型 / ${NUM_GPUS} 卡 (NO_CHAT_TEMPLATE=${NO_CHAT_TEMPLATE})"
for g in "${!GPU_PID[@]}"; do GPU_PID[$g]=0; done
for idx in "${!EVAL_NAMES[@]}"; do
    NAME="${EVAL_NAMES[$idx]}"
    MODEL_PATH="${MODELS_INPUT[$idx]}"
    OUT_BASE="${EVAL_BASES[$idx]}"

    ARGS="pretrained=${MODEL_PATH}"

    mkdir -p "${OUT_BASE}/logs"

    acquire_gpu
    GPU_ID=$ACQUIRED_GPU
    echo ">>> [GPU ${GPU_ID}] ${NAME} → ${MODEL_PATH}  (log: ${OUT_BASE}/logs/*.log)"
    (
        run_gen_math500 0.0 gen_temp0_nomt "${ARGS}" "${OUT_BASE}" "${GPU_ID}"
        for ((j=1; j<=GEN_T1_REPEATS; j++)); do
            run_gen_math500 1.0 "gen_temp1_run${j}_nomt" "${ARGS}" "${OUT_BASE}" "${GPU_ID}"
        done
    ) &
    GPU_PID[${GPU_ID}]=$!
    sleep "${LAUNCH_STAGGER}"
done
wait
echo "[run_models_math500] math500 阶段完成"

# ============================================================================
# 阶段 2: 汇总
# ============================================================================
SUMMARY_CSV="${OUTPUT_ROOT}/eval_summary_math500_${DATE_TAG}.csv"
SUMMARY_JSON="${OUTPUT_ROOT}/eval_summary_math500_${DATE_TAG}.json"

PARENTS=()
for OUT_BASE in "${EVAL_BASES[@]}"; do
    PARENTS+=("$(dirname "${OUT_BASE}")")
done

python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${EVAL_SUBDIR}" "${PARENTS[@]}" <<'PY'
import csv, json, os, sys

csv_path, json_path, eval_subdir, *model_paths = sys.argv[1:]

PREFERRED = [
    "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,none", "acc,none",
]
SKIP_SUBSTR = ("stderr", "sample_len", "alias")

def pick_primary(metrics):
    for k in PREFERRED:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            return k, v
    for k, v in metrics.items():
        if any(s in k for s in SKIP_SUBSTR):
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
            split = (top[:-5] if top.endswith(".json") else top).removeprefix("results_")
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

print("\n========== math500 Eval Summary ==========")
print(f"{'model':<55} {'split':<24} {'task':<20} {'metric':<30} {'value':>8}")
for r in rows:
    print(f"{r['model']:<55} {r['split']:<24} {r['task']:<20} {r['metric']:<30} {r['value']:>8.4f}")
print(f"\n[summary] CSV  → {csv_path}")
print(f"[summary] JSON → {json_path}")
PY

echo "[run_models_math500] 全部完成! 结果在 ${OUTPUT_ROOT}/<model>/${EVAL_SUBDIR}/"
