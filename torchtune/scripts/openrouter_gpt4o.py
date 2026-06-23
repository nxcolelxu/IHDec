import json
import asyncio
import argparse
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm as atqdm


async def call_gpt4o(session, messages, api_key, max_retries=5, retry_delay=5.0):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-4o",
        "messages": messages,
    }
    for attempt in range(max_retries):
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"\n[Attempt {attempt+1}/{max_retries}] Error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
    raise RuntimeError(f"Failed after {max_retries} retries")


async def process_file(src_path, dst_path, api_key, concurrency, save_lock):
    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    # Resume support
    output = [None] * len(data)
    if dst_path.exists():
        with open(dst_path, encoding="utf-8") as f:
            saved = json.load(f)
        if len(saved) == len(data) and saved[0] and saved[0][-1]["role"] == "assistant":
            output = saved

    todo = [i for i, s in enumerate(output) if s is None or s[-1]["role"] != "assistant"]
    print(f"{src_path.name}: {len(todo)} samples to process")

    sem = asyncio.Semaphore(concurrency)
    pbar = atqdm(total=len(todo), desc=src_path.name)

    async def process_one(i, sample, session):
        async with sem:
            messages = [{"role": m["role"], "content": m["content"]} for m in sample]
            reply = await call_gpt4o(session, messages, api_key)
            output[i] = sample + [{"role": "assistant", "content": reply}]
            async with save_lock:
                with open(dst_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=4)
            pbar.update(1)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_one(i, data[i], session) for i in todo]
        await asyncio.gather(*tasks)

    pbar.close()
    print(f"Saved: {dst_path}")


async def main_async(args):
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / args.src_dir
    dst_dir = repo_root / args.dst_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.json"))
    print(f"Found {len(files)} files: {[f.name for f in files]}")

    save_lock = asyncio.Lock()
    for fpath in files:
        await process_file(fpath, dst_dir / fpath.name, args.api_key, args.concurrency, save_lock)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="OpenRouter API key")
    parser.add_argument(
        "--src_dir",
        default="torchtune/data/iheval_ignorance",
        help="Source directory (relative to repo root)",
    )
    parser.add_argument(
        "--dst_dir",
        default="torchtune/data/iheval_ignorance",
        help="Output directory (overwrites src by default)",
    )
    parser.add_argument("--concurrency", type=int, default=20, help="Parallel API calls")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
