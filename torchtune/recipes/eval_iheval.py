#!/usr/bin/env python3
"""
IHEval evaluation for IHDec responses — rule-following and safety domains.

Loads an ihdec.py output file and evaluates it against the IHEval benchmark.
Requires the IHEval repository (https://github.com/microsoft/IHEval).

Dataset names:
    Rule-following: both_conflict_default_request
                    first_conflict_default_request
                    single_conflict_request
    Safety:         hijack_strong, hijack_weak
                    extract_strong, extract_weak

Usage:
    python eval_iheval.py \\
        --input    ../output/iheval_ihdec/both_conflict_default_request_beta1300_decay0.97.json \\
        --dataset  both_conflict_default_request \\
        --iheval-dir ../IHEval

    python eval_iheval.py \\
        --input    ../output/iheval_ihdec/hijack_weak_beta1300_decay0.97.json \\
        --dataset  hijack_weak \\
        --iheval-dir ../IHEval
"""

import argparse
import json
import sys
from pathlib import Path

# ── Dataset → benchmark path mapping ─────────────────────────────────────────

RULE_FOLLOWING_DATASETS = {
    "both_conflict_default_request":  "rule-following/multi-turn/conflict/both-turn-conflict-default-system-prompt",
    "first_conflict_default_request": "rule-following/multi-turn/conflict/first-turn-conflict-default-system-prompt",
    "single_conflict_request":        "rule-following/single-turn/conflict/default-system-prompt",
}

SAFETY_DATASETS = {
    "hijack_strong":  "safety/user-prompt-hijack/conflict/strong_defense",
    "hijack_weak":    "safety/user-prompt-hijack/conflict/weak_defense",
    "extract_strong": "safety/system-prompt-extract/conflict/strong_defense",
    "extract_weak":   "safety/system-prompt-extract/conflict/weak_defense",
}

ALL_DATASETS = {**RULE_FOLLOWING_DATASETS, **SAFETY_DATASETS}


# ── Response loading ──────────────────────────────────────────────────────────

def load_responses(path: Path) -> dict:
    """ihdec.py output JSON → {last_user_message: response}"""
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for item in data:
        response = item.get("response", "")
        if not response:
            continue
        for m in reversed(item.get("conversation", [])):
            if m["role"] == "user":
                result[m["content"]] = response
                break
    return result


# ── Rule-following evaluation ─────────────────────────────────────────────────

def _check_strict(response, inst_ids, kw_list, prompt, reg):
    results = []
    for idx, inst_id in enumerate(inst_ids):
        inst = reg.INSTRUCTION_DICT[inst_id](inst_id)
        inst.build_description(**kw_list[idx])
        if inst.get_instruction_args() and "prompt" in inst.get_instruction_args():
            inst.build_description(prompt=prompt)
        results.append(bool(response.strip() and inst.check_following(response)))
    return results


def _check_loose(response, inst_ids, kw_list, prompt, reg):
    r = response.split("\n")
    variants = [
        response,
        response.replace("*", ""),
        "\n".join(r[1:]).strip(),
        "\n".join(r[:-1]).strip(),
        "\n".join(r[1:-1]).strip(),
        "\n".join(r[1:]).strip().replace("*", ""),
        "\n".join(r[:-1]).strip().replace("*", ""),
        "\n".join(r[1:-1]).strip().replace("*", ""),
    ]
    results = []
    for idx, inst_id in enumerate(inst_ids):
        inst = reg.INSTRUCTION_DICT[inst_id](inst_id)
        inst.build_description(**kw_list[idx])
        if inst.get_instruction_args() and "prompt" in inst.get_instruction_args():
            inst.build_description(prompt=prompt)
        results.append(any(v.strip() and inst.check_following(v) for v in variants))
    return results


def eval_rule_following(response_map, bench_data, iheval_root, log):
    sys.path.insert(0, str(iheval_root / "src" / "rule_following" / "evaluate"))
    import instructions_registry as reg  # noqa: E402

    matched = strict_p = loose_p = strict_i = loose_i = total_i = 0
    details = []

    for i, ex in enumerate(bench_data):
        prompt   = ex["instruction"]
        inst_ids = ex["answer"]["instruction_id_list"]
        kw_list  = ex["answer"]["kwargs"]

        if prompt not in response_map:
            continue
        matched += 1
        response = response_map[prompt]

        sl = _check_strict(response, inst_ids, kw_list, prompt, reg)
        ll = _check_loose(response, inst_ids, kw_list, prompt, reg)

        if all(sl): strict_p += 1
        if all(ll): loose_p  += 1
        strict_i += sum(sl)
        loose_i  += sum(ll)
        total_i  += len(inst_ids)
        details.append({"id": ex["id"], "strict": sl, "loose": ll})

    if matched == 0:
        log("  [!] No matching examples found.")
        return {}

    log(f"  Matched : {matched}")
    log(f"  [Strict]  prompt={strict_p/matched:.1%}  instruction={strict_i/total_i:.1%}")
    log(f"  [Loose]   prompt={loose_p/matched:.1%}  instruction={loose_i/total_i:.1%}")
    log(f"\n  Per-example (strict):")
    for d in details:
        log(f"    [{'O' if all(d['strict']) else 'X'}] id={d['id']:4d}  {d['strict']}")

    return {
        "type": "rule_following", "matched": matched,
        "strict_prompt": strict_p / matched, "loose_prompt": loose_p / matched,
        "strict_inst": strict_i / total_i,  "loose_inst":  loose_i / total_i,
    }


# ── Safety evaluation ─────────────────────────────────────────────────────────

def eval_safety(response_map, bench_data, iheval_root, log):
    sys.path.insert(0, str(iheval_root / "src" / "safety" / "evaluate"))
    from eval_tensortrust import eval_tensortrust  # noqa: E402

    matched = correct = 0
    for ex in bench_data:
        instr = ex["instruction"]
        if instr not in response_map:
            continue
        matched += 1
        if eval_tensortrust(ex["answer"], response_map[instr]):
            correct += 1

    if matched == 0:
        log("  [!] No matching examples found.")
        return {}

    acc = correct / matched
    log(f"  Matched : {matched}  Accuracy: {correct}/{matched} = {acc:.1%}")
    return {"type": "safety", "matched": matched, "accuracy": acc}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",       required=True,
                        help="ihdec.py output JSON file to evaluate")
    parser.add_argument("--dataset",     required=True, choices=list(ALL_DATASETS),
                        help="IHEval dataset name")
    parser.add_argument("--iheval-dir",  default=str(Path(__file__).resolve().parent.parent / "IHEval"),
                        help="Path to IHEval repository root (default: ../IHEval)")
    parser.add_argument("--output",      default=None,
                        help="Optional path to save the evaluation log")
    args = parser.parse_args()

    iheval_root = Path(args.iheval_dir)
    bench_path  = iheval_root / "benchmark" / ALL_DATASETS[args.dataset] / "input_data.json"

    if not bench_path.exists():
        print(f"[ERROR] Benchmark file not found: {bench_path}")
        sys.exit(1)

    bench_data   = json.loads(bench_path.read_text(encoding="utf-8"))
    response_map = load_responses(Path(args.input))

    lines = []
    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log(f"Dataset  : {args.dataset}")
    log(f"Input    : {args.input}")
    log(f"Benchmark: {bench_path}  ({len(bench_data)} examples)")
    log(f"Responses: {len(response_map)} loaded")
    log("=" * 60)

    if args.dataset in RULE_FOLLOWING_DATASETS:
        eval_rule_following(response_map, bench_data, iheval_root, log)
    else:
        eval_safety(response_map, bench_data, iheval_root, log)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nLog saved → {out}")


if __name__ == "__main__":
    main()
