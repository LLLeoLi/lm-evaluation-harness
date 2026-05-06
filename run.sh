#!/bin/bash
set +e
set -u

ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"
NUM_GPUS=8
export HF_HOME="${ENTRY_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1   # 完全靠本地 cache; 先跑 scripts/download_dataset.sh 把数据备齐
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# 清理上次中断遗留的 *.incomplete 目录
find "${HF_DATASETS_CACHE}" -type d -name "*.incomplete" -exec rm -rf {} + 2>/dev/null

# 训练脚本把产物放到 sf_ckpts/${MODEL_BASE}/${MODEL_BASE}-vector-...
# 切换被测 family: CKPT_BASE=Qwen3-8B bash run.sh
CKPT_BASE="${CKPT_BASE:-Llama-3-8B-Instruct}"
CKPT_ROOT="/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts/${CKPT_BASE}-0506"
CKPT_GLOB="${CKPT_GLOB:-${CKPT_BASE}-vector-*-PKU_UnSafeRLHF_100}"

DATE_TAG=$(date +%Y%m%d)
EVAL_SUBDIR="eval_general_${DATE_TAG}"

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
    OUT_DIR="${MODEL_PATH}/${EVAL_SUBDIR}"
    mkdir -p "${OUT_DIR}/logs"

    acquire_gpu
    GPU_ID=$ACQUIRED_GPU
    echo ">>> [GPU ${GPU_ID}] ${NAME} → $(basename ${EVAL_PATH})  (log: ${OUT_DIR}/logs/*.log)"
    (
        run_one 0.0 temp0      "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
        run_one 1.0 temp1_run1 "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
        run_one 1.0 temp1_run2 "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
        run_one 1.0 temp1_run3 "${ARGS}" "${OUT_DIR}" "${GPU_ID}"
    ) &
    GPU_PID[${GPU_ID}]=$!
    sleep "${LAUNCH_STAGGER}"
done

wait

# ============ 汇总: 把所有模型每个 task 的主指标拢到一张表 ============
SUMMARY_CSV="${CKPT_ROOT}/eval_summary_${DATE_TAG}.csv"
SUMMARY_JSON="${CKPT_ROOT}/eval_summary_${DATE_TAG}.json"

python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${EVAL_SUBDIR}" "${MODELS[@]}" <<'PY'
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
    # 兜底: 第一个非 stderr 的数值
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
        # 跳 logs/ 和 samples_*.jsonl
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
            # split: 用 fn 主名 (results_temp0.json -> temp0); 若 lm_eval 把 output_path 当目录,
            # 取 eval_dir 下第一级目录名作为 split
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

echo "[run] 全部完成! 结果在各模型目录下 ${EVAL_SUBDIR}/"
