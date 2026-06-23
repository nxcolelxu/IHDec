#!/usr/bin/env python3
"""
Convert flat per-turn output files to per-conversation JSONL for MT-Bench-101 judge.

Input:
  - v5_eval:      output/v5_eval/mt_bench_101_part{0-3}_beta1300_decay0.97.json
  - instruct_eval: output/instruct_eval/mt_bench_101_part{0-3}/llama3_1_8B_Instruct_consolidated_res.json

Output (JSONL, one line per conversation):
  {task, id, history:[{user,bot},...], model_responses:[str,...]}

Usage:
    python convert_to_mtbench_judge.py --model v5      # → output/judge_input/v5_beta1300_decay0.97.jsonl
    python convert_to_mtbench_judge.py --model instruct # → output/judge_input/instruct.jsonl
    python convert_to_mtbench_judge.py --model both     # both
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

_ROOT        = Path(__file__).resolve().parent.parent
_MTBENCH_SRC = Path.home() / "mt-bench-101/data/subjective/mtbench101.jsonl"
_V5_FULL     = _ROOT / "data/mt_bench_101_v5_full.json"
_OUT_DIR     = _ROOT / "output/judge_input"


def load_metadata() -> list[dict]:
    """Load v5_full.json: [{conversation, task, id, turn}, ...] in sequential order."""
    return json.loads(_V5_FULL.read_text())


def load_reference_history() -> dict[int, list[dict]]:
    """Return {item_id: [{user, bot}, ...]} from original mtbench101.jsonl."""
    result = {}
    for line in _MTBENCH_SRC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        result[item["id"]] = item["history"]
    return result


def load_v5_responses() -> list[str]:
    """Load all v5_eval responses in sequential order across parts 0-3."""
    responses = []
    for i in range(4):
        path = _ROOT / f"output/v5_eval/mt_bench_101_part{i}_beta1300_decay0.97.json"
        part = json.loads(path.read_text())
        part.sort(key=lambda x: x["id"])
        responses.extend(r["response"] for r in part)
    return responses


def load_instruct_responses() -> list[str]:
    """Load all instruct_eval responses in sequential order across parts 0-3."""
    responses = []
    for i in range(4):
        path = _ROOT / f"output/instruct_eval/mt_bench_101_part{i}/llama3_1_8B_Instruct_consolidated_res.json"
        part = json.loads(path.read_text())
        responses.extend(r["response"] for r in part)
    return responses


def build_jsonl(responses: list[str], meta: list[dict], ref_history: dict[int, list[dict]]) -> list[dict]:
    """Group flat per-turn responses into per-conversation entries."""
    assert len(responses) == len(meta), f"{len(responses)} != {len(meta)}"

    # Group by (task, item_id) preserving turn order
    conv_map: dict[tuple, dict] = {}
    for response, m in zip(responses, meta):
        key = (m["task"], m["id"])
        if key not in conv_map:
            conv_map[key] = {
                "task": m["task"],
                "id": m["id"],
                "history": ref_history[m["id"]],
                "model_responses": [],
            }
        conv_map[key]["model_responses"].append(response)

    # Sort by item id for deterministic output
    return sorted(conv_map.values(), key=lambda x: x["id"])


def write_jsonl(items: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(items)} conversations → {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["v5", "instruct", "both"], default="both")
    args = parser.parse_args()

    meta = load_metadata()
    ref_history = load_reference_history()
    print(f"Metadata: {len(meta)} turns, {len(ref_history)} conversations")

    if args.model in ("v5", "both"):
        print("Loading v5 responses...")
        v5_resp = load_v5_responses()
        items = build_jsonl(v5_resp, meta, ref_history)
        write_jsonl(items, _OUT_DIR / "v5_beta1300_decay0.97.jsonl")

    if args.model in ("instruct", "both"):
        print("Loading instruct responses...")
        inst_resp = load_instruct_responses()
        items = build_jsonl(inst_resp, meta, ref_history)
        write_jsonl(items, _OUT_DIR / "instruct.jsonl")


if __name__ == "__main__":
    main()
