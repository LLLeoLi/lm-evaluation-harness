#!/bin/bash
# ==============================================================================
# run_models_v2.sh — 对一组(已存在的) HF 模型直接做 lm-eval-harness 通用能力评测
#
#   设计:
#     - 命令行驱动 (沿用 run_models.sh):
#         bash run_models_v2.sh <model_path_or_id> [<model_path_or_id> ...]
#         MODELS="m1 m2 m3" bash run_models_v2.sh
#         bash run_models_v2.sh -f models.txt        # 每行一个模型
#     - eval 协议沿用 _run_0507_common.sh:
#         * MMLU (loglikelihood): 不加 --apply_chat_template, 不加 --fewshot_as_multiturn,
#                                 用 task 默认 fewshot, 保持与论文报告口径一致.
#         * GSM8K / MATH500 / humaneval_instruct (generate_until):
#             单次 lm_eval 调用, 仅 --apply_chat_template (不加 --fewshot_as_multiturn),
#             T=0 跑 1 次 + T=1 跑 GEN_T1_REPEATS (默认 3) 次.
#         * 含 humaneval 自动加 --confirm_run_unsafe_code + HF_ALLOW_CODE_EVAL=1.
#
#   可选环境变量:
#     LM_TASKS         默认 mmlu                                   (loglikelihood)
#     GEN_TASKS        默认 gsm8k,minerva_math500                  (会自动追加 HUMANEVAL_TASK)
#     HUMANEVAL_TASK   默认 humaneval_instruct (base 模型改 humaneval; 空字符串则不跑)
#     GEN_T1_REPEATS   T=1 重复次数, 默认 1
#     BATCH_SIZE       lm_eval --batch_size, 默认 32
#     NUM_GPUS         可用 GPU 数 (默认按 CUDA_VISIBLE_DEVICES 推断)
#     OUTPUT_ROOT      输出根目录 (默认见下)
#     LIMIT            每任务前 N 道, 调试用 (默认空 = 全量)
#     LAUNCH_STAGGER   多卡启动间隔秒数, 默认 5
#     DTYPE            HF backend dtype, 默认 bfloat16
#
#   AlpacaEval (RUN_ALPACA=1 开启, 在 lm-eval 之前跑; 与 run_models.sh 同款):
#     RUN_ALPACA=1          启用 AlpacaEval (默认 0 关闭)
#     ALPACA_SKIP_JUDGE=1   只用 vLLM 生成 model_outputs.json, 不调 judge
#     ALPACA_JSON           AlpacaEval prompt 集 (默认 ${HF_HOME}/alpaca_eval.json)
#     REFERENCE_JSON        参考输出 (默认 ${HF_HOME}/alpaca_7b_baseline.json)
#     JUDGE_CONFIG_DIR      annotators 配置目录 (默认 alpaca_eval/judge_configs/deepseek_chat)
#     VLLM_PY               生成阶段用的 python (默认 python; 当前 env 需含 vllm + transformers)
#     ALPACA_EVAL_BIN       alpaca_eval CLI 路径 (默认 alpaca_eval)
#     OPENAI_API_BASE       judge LLM 端点 (默认 https://api.deepseek.com)
#     OPENAI_API_KEY        DeepSeek API key (judge 阶段必需; 未设则自动跳过 judge)
#     GEN_TEMP/GEN_TOP_P/GEN_MAX_TOKENS  alpaca 生成采样参数 (默认 0.7 / 1.0 / 512)
#     ALPACA_LIMIT          alpaca 前 N 条 (调试用, 与 lm-eval 的 LIMIT 独立)
#
#   输出:
#     ${OUTPUT_ROOT}/<basename(model)>/${EVAL_SUBDIR}/results_{lm,gen_temp0,gen_temp1_run{1..N}}.json
#     ${OUTPUT_ROOT}/<basename(model)>/${ALPACA_SUBDIR}/{model_outputs.json,judge/leaderboard.csv}
#     ${OUTPUT_ROOT}/eval_summary_${DATE_TAG}.{csv,json}
# ==============================================================================
set +e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LME_DIR="${SCRIPT_DIR}"

# ===== 可用 GPU 数 =====
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
echo "[run_models_v2] NUM_GPUS=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"

# ===== 缓存与离线 (与 _run_0507_common.sh 一致) =====
export HF_HOME="${HF_HOME:-${LME_DIR}/../hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# humaneval 用 evaluate 库的 code_eval metric, import 时就要此 env var
export HF_ALLOW_CODE_EVAL=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
# 清理上次中断遗留的 *.incomplete
find "${HF_DATASETS_CACHE}" -type d -name "*.incomplete" -exec rm -rf {} + 2>/dev/null

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EVAL_SUBDIR="eval_general_${DATE_TAG}"
ALPACA_SUBDIR="alpaca_${DATE_TAG}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LME_DIR}/eval_results_v2_${DATE_TAG}}"

# ===== 任务集 (0507 协议) =====
LM_TASKS="${LM_TASKS:-mmlu}"
HUMANEVAL_TASK="${HUMANEVAL_TASK-humaneval_instruct}"
GEN_TASKS="${GEN_TASKS:-gsm8k,minerva_math500}"
[[ -n "${HUMANEVAL_TASK}" ]] && GEN_TASKS="${GEN_TASKS},${HUMANEVAL_TASK}"
GEN_T1_REPEATS="${GEN_T1_REPEATS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DTYPE="${DTYPE:-bfloat16}"
LAUNCH_STAGGER="${LAUNCH_STAGGER:-5}"

LIMIT="${LIMIT:-}"
LIMIT_ARGS=()
[[ -n "${LIMIT}" ]] && LIMIT_ARGS=( --limit "${LIMIT}" )

GEN_EXTRA_ARGS=()
[[ "${GEN_TASKS}" == *humaneval* ]] && GEN_EXTRA_ARGS+=( --confirm_run_unsafe_code )

# ===== 解析模型列表 =====
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
    echo "[run_models_v2] 用法: bash $0 <model_path> [<model_path> ...]"
    echo "                或: MODELS=\"m1 m2\" bash $0"
    echo "                或: bash $0 -f models.txt"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[run_models_v2] 共 ${#MODELS_INPUT[@]} 个模型, 输出根目录: ${OUTPUT_ROOT}"
echo "[run_models_v2] LM_TASKS=${LM_TASKS}"
echo "[run_models_v2] GEN_TASKS=${GEN_TASKS}"
echo "[run_models_v2] GEN_T1_REPEATS=${GEN_T1_REPEATS}  BATCH_SIZE=${BATCH_SIZE}  DTYPE=${DTYPE}"
for m in "${MODELS_INPUT[@]}"; do echo "  - $m"; done

cd "${LME_DIR}"

# ===== 预解析 NAME / OUT_BASE (与 run_models.sh 同款去重) =====
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
        echo "[run_models_v2] NOTE: ${MODEL_PATH} 不是本地路径, 当作 HF model id 处理"
    fi

    EVAL_BASES+=("${OUTPUT_ROOT}/${NAME}/${EVAL_SUBDIR}")
    EVAL_NAMES+=("${NAME}")
done

# ===== GPU 调度 =====
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

# ============================================================================
# 阶段 1: AlpacaEval (vLLM 生成 + DeepSeek judge) — 与 _run_0507_alpaca_common.sh 对齐
# ============================================================================
if [ "${RUN_ALPACA:-0}" = "1" ]; then
    ALPACA_JSON="${ALPACA_JSON:-${HF_HOME}/alpaca_eval.json}"
    REFERENCE_JSON="${REFERENCE_JSON:-${HF_HOME}/alpaca_7b_baseline.json}"
    JUDGE_CONFIG_DIR="${JUDGE_CONFIG_DIR:-${LME_DIR}/alpaca_eval/judge_configs/deepseek_chat}"
    GEN_SCRIPT="${LME_DIR}/alpaca_eval/generate_alpaca_0507.py"
    VLLM_PY="${VLLM_PY:-python}"
    ALPACA_EVAL_BIN="${ALPACA_EVAL_BIN:-alpaca_eval}"
    GEN_TEMP="${GEN_TEMP:-0.7}"
    GEN_TOP_P="${GEN_TOP_P:-1.0}"
    GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-512}"
    ALPACA_LIMIT="${ALPACA_LIMIT:-}"
    ALPACA_LIMIT_ARGS=()
    [[ -n "${ALPACA_LIMIT}" ]] && ALPACA_LIMIT_ARGS=( --limit "${ALPACA_LIMIT}" )

    if [ ! -f "${ALPACA_JSON}" ]; then
        echo "[alpaca] ERROR: ${ALPACA_JSON} 不存在; 跳过 AlpacaEval"
    elif [ ! -f "${GEN_SCRIPT}" ]; then
        echo "[alpaca] ERROR: 找不到生成脚本 ${GEN_SCRIPT}; 跳过 AlpacaEval"
    else
        # judge 端点 (DeepSeek). 若未 export OPENAI_API_KEY 自动跳过 judge.
        export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.deepseek.com}"
        DO_JUDGE=1
        if [ "${ALPACA_SKIP_JUDGE:-0}" = "1" ]; then
            DO_JUDGE=0
            echo "[alpaca] ALPACA_SKIP_JUDGE=1, 只生成不 judge"
        elif [ -z "${OPENAI_API_KEY:-}" ]; then
            DO_JUDGE=0
            echo "[alpaca] WARN: 未设置 OPENAI_API_KEY, 跳过 judge 阶段 (生成仍会跑)"
        fi

        echo ""
        echo ">>> AlpacaEval Phase 1: vLLM generate (${#EVAL_NAMES[@]} 模型 / ${NUM_GPUS} 卡)"
        echo "    ALPACA_JSON     = ${ALPACA_JSON}"
        echo "    JUDGE_CONFIG    = ${JUDGE_CONFIG_DIR}"
        echo "    GEN sampling    = T=${GEN_TEMP} top_p=${GEN_TOP_P} max=${GEN_MAX_TOKENS}"

        for g in "${!GPU_PID[@]}"; do GPU_PID[$g]=0; done
        for idx in "${!EVAL_NAMES[@]}"; do
            NAME="${EVAL_NAMES[$idx]}"
            MODEL_PATH="${MODELS_INPUT[$idx]}"
            OUT_DIR="${OUTPUT_ROOT}/${NAME}/${ALPACA_SUBDIR}"
            OUT_JSON="${OUT_DIR}/model_outputs.json"
            mkdir -p "${OUT_DIR}/logs"
            # 覆盖语义: 重新生成, 避免拿到旧结果
            rm -f "${OUT_JSON}"

            acquire_gpu
            GPU_ID=$ACQUIRED_GPU
            echo "-> [GPU ${GPU_ID}] alpaca-gen ${NAME}"
            (
                "${VLLM_PY}" "${GEN_SCRIPT}" \
                    --model_path "${MODEL_PATH}" \
                    --output_path "${OUT_JSON}" \
                    --model_name "${NAME}" \
                    --alpaca_json "${ALPACA_JSON}" \
                    --device_ids "${GPU_ID}" \
                    --temperature "${GEN_TEMP}" \
                    --top_p "${GEN_TOP_P}" \
                    --max_tokens "${GEN_MAX_TOKENS}" \
                    "${ALPACA_LIMIT_ARGS[@]}" \
                    > "${OUT_DIR}/logs/gen.log" 2>&1
            ) &
            GPU_PID[${GPU_ID}]=$!
            sleep "${LAUNCH_STAGGER}"
        done
        wait
        echo "[alpaca] 生成阶段完成"

        if [ "${DO_JUDGE}" = "1" ]; then
            echo ""
            echo ">>> AlpacaEval Phase 2: alpaca_eval judge (顺序, 避免 API 限速)"
            for idx in "${!EVAL_NAMES[@]}"; do
                NAME="${EVAL_NAMES[$idx]}"
                OUT_DIR="${OUTPUT_ROOT}/${NAME}/${ALPACA_SUBDIR}"
                OUT_JSON="${OUT_DIR}/model_outputs.json"
                JUDGE_OUT="${OUT_DIR}/judge"
                if [ ! -f "${OUT_JSON}" ]; then
                    echo "[alpaca-skip-judge] ${NAME} (没有 model_outputs.json)"
                    continue
                fi
                # 覆盖语义: 清空 judge 目录, 避免复用旧 leaderboard/标注缓存
                rm -rf "${JUDGE_OUT}"
                mkdir -p "${JUDGE_OUT}"
                echo "-> judge ${NAME}"
                "${ALPACA_EVAL_BIN}" \
                    --model_outputs "${OUT_JSON}" \
                    --reference_outputs "${REFERENCE_JSON}" \
                    --annotators_config "${JUDGE_CONFIG_DIR}" \
                    --output_path "${JUDGE_OUT}" \
                    > "${OUT_DIR}/logs/judge.log" 2>&1
            done
            echo "[alpaca] judge 阶段完成"
        fi
    fi
fi

# ===== 单次 lm_eval 调用 =====
# MMLU: 原始 loglikelihood, 不套 chat template, 不 multiturn
run_lm() {
    local ARGS=$1 OUT_DIR=$2 GPU=$3 NAME=$4
    echo "-> [GPU ${GPU}] ${NAME}  LM (${LM_TASKS})"
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${LM_TASKS}" \
        --device cuda:0 --batch_size "${BATCH_SIZE}" \
        "${LIMIT_ARGS[@]}" \
        --output_path "${OUT_DIR}/results_lm.json" \
        > "${OUT_DIR}/logs/lm.log" 2>&1
}

# GEN: gsm8k + math500 + humaneval_instruct, 单次调用, 仅 --apply_chat_template
run_gen() {
    local TEMP=$1 SUFFIX=$2 ARGS=$3 OUT_DIR=$4 GPU=$5 NAME=$6
    local GEN_KWARGS
    if [ "${TEMP}" = "0.0" ]; then
        GEN_KWARGS="do_sample=False"
    else
        GEN_KWARGS="temperature=${TEMP},do_sample=True"
    fi
    echo "-> [GPU ${GPU}] ${NAME}  GEN ${SUFFIX} (T=${TEMP})"
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "${ARGS}" --tasks "${GEN_TASKS}" \
        --apply_chat_template \
        --device cuda:0 --batch_size "${BATCH_SIZE}" \
        --gen_kwargs "${GEN_KWARGS}" \
        "${GEN_EXTRA_ARGS[@]}" \
        "${LIMIT_ARGS[@]}" \
        --output_path "${OUT_DIR}/results_${SUFFIX}.json" \
        > "${OUT_DIR}/logs/${SUFFIX}.log" 2>&1
}

# ============================================================================
# 阶段 2: lm-eval (mmlu / gsm8k / math500 / humaneval_instruct)
# ============================================================================
echo ""
echo ">>> lm-eval Phase: ${#EVAL_NAMES[@]} 模型 / ${NUM_GPUS} 卡"
# 进入 lm-eval 前 reset GPU 状态 (alpaca 阶段已 wait, 保险起见)
for g in "${!GPU_PID[@]}"; do GPU_PID[$g]=0; done
for idx in "${!EVAL_NAMES[@]}"; do
    NAME="${EVAL_NAMES[$idx]}"
    MODEL_PATH="${MODELS_INPUT[$idx]}"
    OUT_BASE="${EVAL_BASES[$idx]}"

    ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE}"
    # Qwen3 原版 (hybrid thinking) 需关 thinking; Qwen3-*-Instruct-2507 / *_dpo_* 不带此机制
    case "${MODEL_PATH}" in
        *Qwen3*Instruct-2507*) ;;
        *Qwen3*dpo*) ;;
        *Qwen3*) ARGS="${ARGS},enable_thinking=False" ;;
    esac

    mkdir -p "${OUT_BASE}/logs"

    acquire_gpu
    GPU_ID=$ACQUIRED_GPU
    echo ">>> [GPU ${GPU_ID}] ${NAME} → ${MODEL_PATH}  (log: ${OUT_BASE}/logs/*.log)"
    (
        [[ -n "${LM_TASKS}" ]]  && run_lm  "${ARGS}" "${OUT_BASE}" "${GPU_ID}" "${NAME}"
        [[ -n "${GEN_TASKS}" ]] && run_gen 0.0 gen_temp0 "${ARGS}" "${OUT_BASE}" "${GPU_ID}" "${NAME}"
        for ((j=1; j<=GEN_T1_REPEATS; j++)); do
            [[ -n "${GEN_TASKS}" ]] && run_gen 1.0 "gen_temp1_run${j}" "${ARGS}" "${OUT_BASE}" "${GPU_ID}" "${NAME}"
        done
    ) &
    GPU_PID[${GPU_ID}]=$!
    sleep "${LAUNCH_STAGGER}"
done
wait
echo "[run_models_v2] lm-eval 阶段完成"

# ===== 汇总 =====
SUMMARY_CSV="${OUTPUT_ROOT}/eval_summary_${DATE_TAG}.csv"
SUMMARY_JSON="${OUTPUT_ROOT}/eval_summary_${DATE_TAG}.json"
PARENTS=()
for OUT_BASE in "${EVAL_BASES[@]}"; do
    PARENTS+=("$(dirname "${OUT_BASE}")")
done

python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${EVAL_SUBDIR}" "${ALPACA_SUBDIR}" "${PARENTS[@]}" <<'PY'
import csv, json, os, sys

csv_path, json_path, eval_subdir, alpaca_subdir, *model_paths = sys.argv[1:]

PREFERRED = [
    "acc,none", "acc_norm,none",
    "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,none",
    "pass@1,create_test", "pass@1,none",
    "mc2,none", "em,none",
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
                    "model": name, "split": split, "task": task,
                    "metric": metric_key, "value": float(value),
                })

    # --- AlpacaEval leaderboard.csv ---
    leaderboard = os.path.join(mp, alpaca_subdir, "judge", "leaderboard.csv")
    if os.path.isfile(leaderboard):
        try:
            with open(leaderboard, "r", encoding="utf-8") as f:
                lb_rows = list(csv.DictReader(f))
        except Exception as e:
            print(f"[summary] WARN: 读取失败 {leaderboard}: {e}")
            lb_rows = []
        target = None
        for r in lb_rows:
            mid = r.get("") or r.get("model") or r.get("Unnamed: 0") or ""
            if mid == name:
                target = r
                break
        if target is None and lb_rows:
            for r in lb_rows:
                mid = r.get("") or r.get("model") or r.get("Unnamed: 0") or ""
                if "baseline" not in mid.lower() and "gpt4" not in mid.lower():
                    target = r
                    break
        if target is not None:
            for metric_key in ("win_rate", "length_controlled_winrate", "standard_error"):
                v = target.get(metric_key)
                if v in (None, ""):
                    continue
                try:
                    rows.append({
                        "model": name, "split": "alpaca_eval", "task": "alpaca_eval",
                        "metric": metric_key, "value": float(v),
                    })
                except ValueError:
                    pass

rows.sort(key=lambda r: (r["model"], r["split"], r["task"]))

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "split", "task", "metric", "value"])
    for r in rows:
        w.writerow([r["model"], r["split"], r["task"], r["metric"], f"{r['value']:.4f}"])

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("\n========== Eval Summary ==========")
print(f"{'model':<55} {'split':<20} {'task':<25} {'metric':<30} {'value':>8}")
for r in rows:
    print(f"{r['model']:<55} {r['split']:<20} {r['task']:<25} {r['metric']:<30} {r['value']:>8.4f}")
print(f"\n[summary] CSV  → {csv_path}")
print(f"[summary] JSON → {json_path}")
PY

echo "[run_models_v2] 全部完成! 结果在 ${OUTPUT_ROOT}/<model>/${EVAL_SUBDIR}/"
