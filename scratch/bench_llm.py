import time
import httpx

prompt = """You are a professional AI voice assistant handling incoming customer inquiries.
Generate a warm, professional, natural spoken response to the customer's latest statement.

CRITICAL INSTRUCTIONS:
1. Respond directly to the current customer statement. Use prior conversation history for context without repeating everything.
2. NEVER invent, assume, or hallucinate prices, budgets, timelines, product details, or business commitments.
3. If key information (such as budget, timeline, or scope) remains missing after checking both history and the current statement, ask ONE clear, relevant follow-up question.
4. Write in a clear, natural conversational style optimized to sound great when spoken aloud by a TTS system.
5. Do NOT include markdown code blocks, formatting tags, emojis, or JSON. Output plain conversational text only.
6. Keep your response concise (1 to 3 sentences maximum), ideal for voice calls.

Current Customer Statement:
"I need 50 laptops with 16GB RAM for our department next month. Our budget is 45000 dollars."
"""

client = httpx.Client(timeout=120.0)

print("Starting benchmark...")
# Test 1: baseline
t0 = time.perf_counter()
r1 = client.post('http://localhost:11434/api/generate', json={
    'model': 'gemma3:4b',
    'prompt': prompt,
    'stream': False,
    'keep_alive': '30m',
    'options': {'num_predict': 75, 'temperature': 0.7}
})
t1 = time.perf_counter()
d1 = r1.json()
eval_dur1 = round(d1.get('eval_duration', 0) / 1e9, 2)
prompt_dur1 = round(d1.get('prompt_eval_duration', 0) / 1e9, 2)
print(f"Test 1 (baseline): Total={round(t1-t0, 2)}s | PromptEval={prompt_dur1}s | Tokens={d1.get('eval_count')} | Generation={eval_dur1}s")
print(f"Output 1: {repr(d1.get('response').strip())}")

# Test 2: with num_thread=8 and stop tokens
t0 = time.perf_counter()
r2 = client.post('http://localhost:11434/api/generate', json={
    'model': 'gemma3:4b',
    'prompt': prompt,
    'stream': False,
    'keep_alive': '30m',
    'options': {'num_predict': 60, 'temperature': 0.7, 'num_thread': 8, 'stop': ['\n\n', 'Customer:']}
})
t2 = time.perf_counter()
d2 = r2.json()
eval_dur2 = round(d2.get('eval_duration', 0) / 1e9, 2)
prompt_dur2 = round(d2.get('prompt_eval_duration', 0) / 1e9, 2)
print(f"Test 2 (num_thread=8): Total={round(t2-t0, 2)}s | PromptEval={prompt_dur2}s | Tokens={d2.get('eval_count')} | Generation={eval_dur2}s")
print(f"Output 2: {repr(d2.get('response').strip())}")

# Test 3: with num_thread=12 and stop tokens
t0 = time.perf_counter()
r3 = client.post('http://localhost:11434/api/generate', json={
    'model': 'gemma3:4b',
    'prompt': prompt,
    'stream': False,
    'keep_alive': '30m',
    'options': {'num_predict': 50, 'temperature': 0.7, 'num_thread': 12, 'stop': ['\n\n', 'Customer:']}
})
t3 = time.perf_counter()
d3 = r3.json()
eval_dur3 = round(d3.get('eval_duration', 0) / 1e9, 2)
prompt_dur3 = round(d3.get('prompt_eval_duration', 0) / 1e9, 2)
print(f"Test 3 (num_thread=12, 50 tok): Total={round(t3-t0, 2)}s | PromptEval={prompt_dur3}s | Tokens={d3.get('eval_count')} | Generation={eval_dur3}s")
print(f"Output 3: {repr(d3.get('response').strip())}")
