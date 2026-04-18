import httpx
from config import settings

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_MODEL = "llama-3.1-8b-instant"


class GroqProvider:
    def __init__(self) -> None:
        self.api_key = settings.groq_api_key
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set")

    async def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                _GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": _DEFAULT_MODEL, "messages": messages},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        # Groq doesn't offer an embedding endpoint — delegate to Ollama
        from services.llm.ollama import OllamaProvider
        return await OllamaProvider().embed(text)
