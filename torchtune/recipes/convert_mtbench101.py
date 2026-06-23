#!/usr/bin/env python3
"""
Convert MT-Bench-101 (mtbench101.jsonl) to the conversation format used by
ihdec.py and evaluate.py.

Output format (no system prompt):
  [
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...,
     {"role": "user", "content": "last question"}],
    ...
  ]

Usage:
    python convert_mtbench101.py
    python convert_mtbench101.py --src /path/to/mtbench101.jsonl
"""

import argparse
import json
from pathlib import Path

_DEFAULT_SRC = Path.home() / "mt-bench-101/data/subjective/mtbench101.jsonl"
_DATA_DIR    = Path(__file__).resolve().parent.parent / "data"


def convert(src: Path) -> list:
    """One input item per turn, context built from reference bot answers.

    For a 3-turn conversation:
      item 0: [user1]
      item 1: [user1, bot1(ref), user2]
      item 2: [user1, bot1(ref), user2, bot2(ref), user3]
    """
    items = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    result = []
    for item in items:
        history = item["history"]
        context = []
        for turn in history:
            conv = context + [{"role": "user", "content": turn["user"]}]
            result.append({
                "conversation": conv,
                "task": item["task"],
                "id": item["id"],
                "turn": len(context) // 2 + 1,
            })
            context.append({"role": "user",      "content": turn["user"]})
            context.append({"role": "assistant", "content": turn["bot"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=str, default=str(_DEFAULT_SRC))
    args = parser.parse_args()

    data = convert(Path(args.src))
    print(f"Total samples: {len(data)}")

    conversations = [item["conversation"] for item in data]

    out = _DATA_DIR / "mt_bench_101.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
