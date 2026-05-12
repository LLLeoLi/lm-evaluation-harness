#!/usr/bin/env python3
"""
summarize_eval.py — 汇总 run_models.sh 的输出, 按
  MMLU / AlpacaEval / GSM8K / MATH / humaneval
列序生成表 (markdown + csv + json).

用法:
  python summarize_eval.py <OUTPUT_ROOT> [--date YYYYMMDD] [--out PREFIX]

约定的目录结构 (与 run_models.sh 一致):
  <OUTPUT_ROOT>/<model_name>/eval_general_<DATE>/results_lm.json
  <OUTPUT_ROOT>/<model_name>/eval_general_<DATE>/results_gen_temp0.json
  <OUTPUT_ROOT>/<model_name>/alpaca_<DATE>/judge/leaderboard.csv

也可以直接给一个父目录里只放各模型子目录的情形 — 脚本会自动从子目录里挑最新的
eval_general_* / alpaca_* 子目录。
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

# 每个目标基准对应的 (raw result 文件 stem, lm-eval task name, metric key 优先级)
BENCH_SPEC = [
    ("MMLU",       "results_lm",        "mmlu",
        ["acc,none"]),
    ("GSM8K",      "results_gen_temp0", "gsm8k",
        ["exact_match,strict-match", "exact_match,flexible-extract"]),
    ("MATH",       "results_gen_temp0", "hendrycks_math",
        ["exact_match,none", "exact_match,strict-match", "exact_match,flexible-extract"]),
    ("humaneval",  "results_gen_temp0", "humaneval_instruct",
        ["pass@1,create_test", "pass@1,none"]),
    # humaneval base 模型 fallback
    ("humaneval",  "results_gen_temp0", "humaneval",
        ["pass@1,create_test", "pass@1,none"]),
]


def latest_subdir(model_dir, prefix):
    """挑选 <model_dir>/<prefix>_<DATE>/ 里最新的一个 (按目录名排序)."""
    cands = sorted(glob.glob(os.path.join(model_dir, f"{prefix}_*")))
    return cands[-1] if cands else None


def load_results(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("results", {})
    except Exception as e:
        print(f"[warn] read {path}: {e}", file=sys.stderr)
        return {}


def pick(metrics, keys):
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            return v
    return None


def gather_lm_metrics(model_dir, date_filter):
    """合并 eval_general_<DATE>/ 下所有 results 文件 (按 date_filter 过滤).

    注意: lm_eval 把 --output_path .../results_lm.json 当作目录前缀,
    真实文件是 .../results_lm.json/<sanitized_path>/results_<timestamp>.json
    所以这里递归遍历, 用相对路径前缀 (results_lm vs results_gen_temp0) 判定 split."""
    if date_filter:
        eval_dir = os.path.join(model_dir, f"eval_general_{date_filter}")
        if not os.path.isdir(eval_dir):
            eval_dir = latest_subdir(model_dir, "eval_general")
    else:
        eval_dir = latest_subdir(model_dir, "eval_general")
    merged = {}
    if not (eval_dir and os.path.isdir(eval_dir)):
        return merged, eval_dir
    for root, dirs, files in os.walk(eval_dir):
        if os.path.basename(root) == "logs":
            dirs[:] = []
            continue
        for fn in files:
            if not fn.endswith(".json"):
                continue
            if fn.startswith("samples") or "samples_" in fn:
                continue
            fp = os.path.join(root, fn)
            results = load_results(fp)
            if not results:
                continue
            for task, metrics in results.items():
                if isinstance(metrics, dict):
                    # 后写的覆盖先写的; 同任务多份结果时取后出现的 (一般是最新)
                    merged[task] = metrics
    return merged, eval_dir


def gather_alpaca(model_dir, model_name, date_filter):
    if date_filter:
        alp_dir = os.path.join(model_dir, f"alpaca_{date_filter}")
        if not os.path.isdir(alp_dir):
            alp_dir = latest_subdir(model_dir, "alpaca")
    else:
        alp_dir = latest_subdir(model_dir, "alpaca")
    if not alp_dir:
        return None, None
    lb = os.path.join(alp_dir, "judge", "leaderboard.csv")
    if not os.path.isfile(lb):
        return None, alp_dir
    try:
        with open(lb, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"[warn] read {lb}: {e}", file=sys.stderr)
        return None, alp_dir
    target = None
    for r in rows:
        mid = r.get("") or r.get("model") or r.get("Unnamed: 0") or ""
        if mid == model_name:
            target = r
            break
    if target is None:
        for r in rows:
            mid = (r.get("") or r.get("model") or r.get("Unnamed: 0") or "").lower()
            if mid and "baseline" not in mid and "gpt4" not in mid:
                target = r
                break
    if target is None:
        return None, alp_dir
    for k in ("length_controlled_winrate", "win_rate"):
        v = target.get(k)
        if v not in (None, ""):
            try:
                return float(v), alp_dir
            except ValueError:
                pass
    return None, alp_dir


def collect_row(model_dir, date_filter):
    name = os.path.basename(model_dir.rstrip("/"))
    res, eval_dir = gather_lm_metrics(model_dir, date_filter)
    alpaca, alp_dir = gather_alpaca(model_dir, name, date_filter)

    row = {"model": name}
    # MMLU
    row["MMLU"] = pick(res.get("mmlu", {}), ["acc,none"])
    # AlpacaEval
    row["AlpacaEval"] = alpaca
    # GSM8K
    row["GSM8K"] = pick(res.get("gsm8k", {}),
                        ["exact_match,strict-match", "exact_match,flexible-extract"])
    # MATH
    row["MATH"] = pick(res.get("hendrycks_math", {}),
                       ["exact_match,none", "exact_match,strict-match", "exact_match,flexible-extract"])
    # humaneval (instruct 优先, 否则 base)
    he_metrics = res.get("humaneval_instruct") or res.get("humaneval") or {}
    row["humaneval"] = pick(he_metrics, ["pass@1,create_test", "pass@1,none"])

    row["_eval_dir"] = eval_dir
    row["_alpaca_dir"] = alp_dir
    return row


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root", help="run_models.sh 的 OUTPUT_ROOT 目录")
    ap.add_argument("--date", default=None,
                    help="只取 eval_general_<DATE>/alpaca_<DATE> 这个日期; 默认每个模型挑最新")
    ap.add_argument("--out", default=None,
                    help="输出前缀; 写出 <prefix>.csv / <prefix>.json / <prefix>.md (默认 <OUTPUT_ROOT>/eval_table[_<DATE>])")
    ap.add_argument("--include", default=None,
                    help="只保留 basename 匹配该正则的模型")
    args = ap.parse_args()

    root = os.path.abspath(args.output_root)
    if not os.path.isdir(root):
        print(f"[err] not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    model_dirs = []
    for entry in sorted(os.listdir(root)):
        p = os.path.join(root, entry)
        if not os.path.isdir(p):
            continue
        if entry.startswith("eval_summary") or entry.startswith("."):
            continue
        if args.include and not re.search(args.include, entry):
            continue
        # 必须至少有 eval_general_* 或 alpaca_* 子目录
        if glob.glob(os.path.join(p, "eval_general_*")) or glob.glob(os.path.join(p, "alpaca_*")):
            model_dirs.append(p)

    if not model_dirs:
        print(f"[err] no model subdirs with eval_general_* or alpaca_* under {root}",
              file=sys.stderr)
        sys.exit(1)

    rows = [collect_row(md, args.date) for md in model_dirs]
    cols = ["MMLU", "AlpacaEval", "GSM8K", "MATH", "humaneval"]

    # ---- markdown ----
    name_w = max(len(r["model"]) for r in rows)
    name_w = max(name_w, len("model"))
    header = "| " + "model".ljust(name_w) + " | " + " | ".join(f"{c:>10}" for c in cols) + " |"
    sep    = "|" + "-" * (name_w + 2) + "|" + "|".join(["-" * 12] * len(cols)) + "|"
    md_lines = [header, sep]
    for r in rows:
        md_lines.append(
            "| " + r["model"].ljust(name_w) + " | "
            + " | ".join(f"{fmt(r[c]):>10}" for c in cols) + " |"
        )
    md = "\n".join(md_lines)
    print(md)

    out_prefix = args.out or os.path.join(
        root, "eval_table" + (f"_{args.date}" if args.date else ""))
    with open(out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(out_prefix + ".csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + cols)
        for r in rows:
            w.writerow([r["model"]] + [
                f"{r[c]:.4f}" if isinstance(r[c], (int, float)) else "" for c in cols
            ])
    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(
            [{k: r[k] for k in ["model"] + cols} for r in rows],
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n[written] {out_prefix}.md / .csv / .json", file=sys.stderr)

    missing = []
    for r in rows:
        miss = [c for c in cols if r[c] is None]
        if miss:
            missing.append((r["model"], miss))
    if missing:
        print("\n[note] 缺失项:", file=sys.stderr)
        for name, miss in missing:
            print(f"  {name}: {', '.join(miss)}", file=sys.stderr)


if __name__ == "__main__":
    main()
