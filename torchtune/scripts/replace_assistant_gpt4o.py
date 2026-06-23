"""
single_aligned_request.json의 각 샘플을 GPT-4o에 보내서 응답을 받고,
iheval_ignorance 4개 파일의 assistant 메시지를 그 응답으로 교체한다.
응답은 캐시 파일에 저장해서 중단 후 재시작 가능.
"""

import json
import asyncio
import argparse
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm as atqdm

CACHE_FILE = Path(__file__).resolve().parent / "gpt4o_aligned_cache.json"

TARGET_FILES = [
    "both_conflict_default_request.json",
    "both_conflict_strong_request.json",
    "first_conflict_default_request.json",
    "first_conflict_strong_request.json",
]


async def call_gpt4o(session, messages, api_key, max_retries=5, retry_delay=5.0):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": "openai/gpt-4o", "messages": messages}
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


async def get_responses(aligned_data, api_key, concurrency):
    # Load cache
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    else:
        cache = [None] * len(aligned_data)

    todo = [i for i, v in enumerate(cache) if v is None]
    print(f"GPT-4o 응답 수집: {len(aligned_data) - len(todo)}개 캐시됨, {len(todo)}개 남음")

    if not todo:
        return cache

    sem = asyncio.Semaphore(concurrency)
    save_lock = asyncio.Lock()
    pbar = atqdm(total=len(todo), desc="GPT-4o 응답 수집")

    async def process_one(i, session):
        async with sem:
            messages = [{"role": m["role"], "content": m["content"]} for m in aligned_data[i]]
            reply = await call_gpt4o(session, messages, api_key)
            cache[i] = reply
            async with save_lock:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
            pbar.update(1)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*[process_one(i, session) for i in todo])

    pbar.close()
    return cache


def replace_assistant(ignorance_dir, responses):
    for fname in TARGET_FILES:
        path = ignorance_dir / fname
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for i, sample in enumerate(data):
            for msg in sample:
                if msg["role"] == "assistant":
                    msg["content"] = responses[i]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved: {fname}")


async def main_async(args):
    repo_root = Path(__file__).resolve().parents[2]
    aligned_path = repo_root / "torchtune/data/iheval/single_aligned_request.json"
    ignorance_dir = repo_root / "torchtune/data/iheval_ignorance"

    with open(aligned_path, encoding="utf-8") as f:
        aligned_data = json.load(f)

    responses = await get_responses(aligned_data, args.api_key, args.concurrency)
    replace_assistant(ignorance_dir, responses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
