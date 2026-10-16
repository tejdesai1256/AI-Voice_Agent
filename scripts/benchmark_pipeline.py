import sys
import os
import time
import asyncio

sys.path.insert(0, os.path.abspath("."))

from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.services.llm_service import generate_response

print("=== PROFILING PIPELINE STAGES ===")

# 1. STT
print("1. Profiling STT...")
stt = STTService()
t0 = time.perf_counter()
txt = stt.transcribe("static/waiting/waiting_1.wav")
t_stt = time.perf_counter() - t0
print(f"   STT took: {t_stt:.3f}s | Result: '{txt}'")

# 2. TTS
print("2. Profiling TTS...")
tts = TTSService()
t0 = time.perf_counter()
tts.synthesize("Hello! How can I help you today?", "recordings/bench_tts.wav")
t_tts = time.perf_counter() - t0
print(f"   TTS took: {t_tts:.3f}s")

# 3. LLM (Single Call)
async def profile_llm():
    print("3. Profiling LLM (Single Call)...")
    prompt = (
        "You are a professional AI voice assistant handling incoming customer inquiries.\n"
        "Generate a warm, professional, natural spoken response to the customer's latest statement.\n"
        "Keep your response concise (1 to 2 sentences maximum).\n\n"
        "Current Customer Statement:\n\"Hello, I am calling to inquire about web development services.\"\n"
    )
    t0 = time.perf_counter()
    res = await generate_response(prompt)
    t_llm = time.perf_counter() - t0
    print(f"   LLM took: {t_llm:.3f}s | Result: '{res.strip()}'")

asyncio.run(profile_llm())
