"""Generate alpaca_eval responses with vLLM.

基于 NSPO/evaluation/generate_alpaca.py 改写, 区别:
  1. 直接读 hf_cache/alpaca_eval.json (绕过 datasets script 已 deprecated)
  2. 输出格式与 alpaca-eval CLI 兼容: [{instruction, output, generator, dataset}]
"""
import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", required=True, help="alpaca-eval 里 generator 字段; 也用作 win-rate 报告里的 model id")
    parser.add_argument("--alpaca_json", required=True, help="alpaca_eval.json 路径 (805 instr)")
    parser.add_argument("--device_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条 (调试用)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                        help="vLLM 显存占用比例; GPU 已有其他进程时调低 (默认 0.7)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in args.device_ids)

    with open(args.alpaca_json) as f:
        eval_set = json.load(f)
    if args.limit is not None:
        eval_set = eval_set[: args.limit]
    print(f"[gen] {len(eval_set)} instructions")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=len(args.device_ids),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tok = AutoTokenizer.from_pretrained(args.model_path)

    prompts = []
    for item in eval_set:
        msgs = [{"role": "user", "content": item["instruction"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    outputs = llm.generate(prompts, sp)

    results = []
    for item, out in zip(eval_set, outputs):
        results.append({
            "instruction": item["instruction"],
            "output": out.outputs[0].text,
            "generator": args.model_name,
            "dataset": item.get("dataset", "alpaca_eval"),
        })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[gen] saved → {args.output_path}")


if __name__ == "__main__":
    main()
