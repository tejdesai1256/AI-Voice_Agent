import sys
import os
import time
import asyncio
import httpx

sys.path.insert(0, os.path.abspath("."))

async def test_tokens(num_tokens):
    client = httpx.AsyncClient(timeout=60.0)
    prompt = (
        "You are an AI voice assistant answering an incoming call.\n"
        "Customer says: \"Hello, I want to know what services you provide.\"\n"
        "Respond warmly and naturally in 1 or 2 concise spoken sentences."
    )
    t0 = time.perf_counter()
    r = await client.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_tokens, "temperature": 0.7}
        }
    )
    t_taken = time.perf_counter() - t0
    resp = r.json().get("response", "").strip()
    print(f"num_predict={num_tokens} -> {t_taken:.2f}s | Reply: '{resp}'")

async def main():
    await test_tokens(55)
    await test_tokens(80)

asyncio.run(main())
