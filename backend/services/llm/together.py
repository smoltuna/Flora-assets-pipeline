import httpx
from config import settings

_TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"
_TOGETHER_EMBED_URL = "https://api.together.xyz/v1/embeddings"
_DEFAULT_LLM = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
_DEFAULT_EMBED = "togethercomputer/m2-bert-80M-8k-retrieval"


class TogetherProvider:
    def __init__(self) -> None:
        self.api_key = settings.together_api_key
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY is not set")

    async def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                _TOGETHER_CHAT_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": _DEFAULT_LLM, "messages": messages},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                _TOGETHER_EMBED_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": _DEFAULT_EMBED, "input": text},
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
