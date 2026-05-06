echo "[setup] 安装 lm-evaluation-harness ..."
pip install -e . --user
pip install "lm_eval[hf]" --user
bash /opt/tiger/entry/scripts/download_dataset.sh