#!/bin/bash
# setup_math500.sh — 在 setup.sh 已跑过的环境上, 追加预下载 minerva_math500 用的
# HuggingFaceH4/MATH-500 数据集到 HF datasets cache. run_models_math500.sh 跑
# minerva_math500 时是 HF_DATASETS_OFFLINE=1, 没有提前预热会炸:
#   ConnectionError: Couldn't reach 'HuggingFaceH4/MATH-500' on the Hub
#
# 用法: bash setup_math500.sh
# 重复执行安全: 已 build 过的会跳过.

set -u

ENTRY_DIR="/opt/tiger/entry"
LME_DIR="${ENTRY_DIR}/lm-evaluation-harness"
HF_HOME_DIR="${ENTRY_DIR}/hf_cache"

mkdir -p "${HF_HOME_DIR}"

# 下载阶段必须联网, 显式清掉 setup.sh 外可能继承的 offline flag
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME="${HF_HOME_DIR}"
export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"

echo "================================================="
echo "[setup_math500] HF_HOME = ${HF_HOME}"
echo "[setup_math500] python  = $(which python)"
echo "================================================="

# minerva_math500 任务用的就这一个 dataset
DATASETS=(
    "HuggingFaceH4/MATH-500"
)

for entry in "${DATASETS[@]}"; do
    repo="${entry%%:*}"
    cfgs="${entry#*:}"
    [ "${cfgs}" = "${entry}" ] && cfgs=""
    builder_name="${repo//\//___}"
    if [ -d "${HF_DATASETS_CACHE}/${builder_name}" ]; then
        echo "[skip] dataset ${repo} 已有 builder cache"
        continue
    fi
    if [ -z "${cfgs}" ]; then
        echo "[download] dataset ${repo} ..."
        python -c "from datasets import load_dataset; load_dataset('${repo}'); print('[ok] ${repo}')" \
            || { echo "[ERROR] 下载/build ${repo} 失败"; exit 1; }
    else
        IFS=',' read -r -a cfg_arr <<< "${cfgs}"
        for cfg in "${cfg_arr[@]}"; do
            echo "[download] dataset ${repo} (config=${cfg}) ..."
            python -c "from datasets import load_dataset; load_dataset('${repo}', '${cfg}'); print('[ok] ${repo}/${cfg}')" \
                || { echo "[ERROR] 下载/build ${repo}/${cfg} 失败"; exit 1; }
        done
    fi
done

echo ""
echo "================================================="
echo "[setup_math500] 完成. 现在可以离线跑 minerva_math500:"
echo "  bash run_models_math500.sh -f models.txt"
echo "================================================="
