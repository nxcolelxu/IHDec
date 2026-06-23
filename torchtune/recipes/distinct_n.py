"""
Compute Distinct-1/2/3 for response JSON files using a HuggingFace tokenizer.

Usage:
    python distinct_n.py --tokenizer <path> --input <file1> [<file2> ...]

Examples:
    # Qwen ablation_norm files
    python distinct_n.py \
        --tokenizer ../pretrained_models/Qwen3-8B-Instruct \
        --input ../output/ablation_norm/Qwen3-8B-v5-both-conflict-default.json

    # Llama llama_v5_n file
    python distinct_n.py \
        --tokenizer ../pretrained_models/llama3_1_8B_Instruct \
        --input ../data/ih-jsd-bias-responses/llama_v5_n/both_conflict_default_request.json_beta1300.0_decay0.97.json
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def distinct_n(sentences, n):
    ngrams = []
    for sent in sentences:
        ngrams.extend(tuple(sent[i:i+n]) for i in range(len(sent) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute(tokenizer, path):
    data = json.load(open(path, encoding="utf-8"))
    responses = [r["response"] for r in data if r.get("response")]
    tokens = [tokenizer.encode(r, add_special_tokens=False) for r in responses]
    return {
        "n": len(tokens),
        "d1": distinct_n(tokens, 1),
        "d2": distinct_n(tokens, 2),
        "d3": distinct_n(tokens, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer path or name")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSON file(s)")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"\n{'파일명':<65} {'N':>5} {'D-1':>7} {'D-2':>7} {'D-3':>7}")
    print("-" * 95)
    for path in args.input:
        r = compute(tokenizer, path)
        print(f"{Path(path).name:<65} {r['n']:>5} {r['d1']:>7.4f} {r['d2']:>7.4f} {r['d3']:>7.4f}")


if __name__ == "__main__":
    main()
