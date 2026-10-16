import asyncio
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

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


async def generate_response(prompt: str, num_predict: int = 75) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_predict": num_predict,
            "temperature": 0.7,
        },
    }

    client = get_client()
    try:
        response = await client.post(
            OLLAMA_URL,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["response"]
    except httpx.ConnectError:
        raise RuntimeError(
            f"Ollama server is unreachable at {OLLAMA_URL}. "
            f"Please verify that Ollama is running ('ollama serve') and model '{MODEL_NAME}' is pulled ('ollama pull {MODEL_NAME}')."
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Ollama returned HTTP error {e.response.status_code}: {e.response.text}. "
            f"Make sure model '{MODEL_NAME}' is installed via 'ollama pull {MODEL_NAME}'."
        )
