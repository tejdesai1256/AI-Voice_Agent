import os
import asyncio
import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

_client: httpx.AsyncClient = None
_client_loop: asyncio.AbstractEventLoop = None


def get_client() -> httpx.AsyncClient:
    global _client, _client_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if (
        _client is None
        or _client.is_closed
        or _client_loop is None
        or _client_loop.is_closed()
        or _client_loop is not current_loop
    ):
        _client = httpx.AsyncClient(timeout=120.0)
        _client_loop = current_loop
    return _client


async def generate_response(prompt: str, num_predict: int = 45, model: str = None) -> str:
    """
    Generates LLM text response via local Ollama API.
    Options:
    - num_predict: Token budget (default: 45 for ultra-fast, natural spoken responses)
    - num_gpu: 0 (forces CPU execution to prevent Windows CUDA stack buffer overrun)
    """
    active_model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL_NAME)
    active_url = os.environ.get("OLLAMA_URL", OLLAMA_URL)

    payload = {
        "model": active_model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_gpu": 0,
            "num_predict": num_predict,
            "temperature": 0.7,
            "num_ctx": 2048,
        },
    }

    client = get_client()
    try:
        response = await client.post(
            active_url,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException, Exception) as err:
        print(f"[LLM Warning] Ollama request failed ({err}). Evaluating safe fallback.")
        # If this was requirement extraction (asking for JSON)
        if "valid JSON object" in prompt or "requirement extraction" in prompt:
            return '{"requirements": null, "products_or_services": null, "quantity": null, "budget": null, "deadline": null, "important_details": []}'
        # If this was conversational speech, return a natural spoken fallback
        return "Thank you for letting me know. I have noted that down. Could you share any other specific details or requirements you have?"
