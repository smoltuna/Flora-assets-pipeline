from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, prompt: str, system: str = "") -> str: ...
    async def embed(self, text: str) -> list[float]: ...


def get_provider(provider_name: str | None = None) -> LLMProvider:
    from config import settings
    name = provider_name or settings.llm_provider

    if name == "groq":
        from services.llm.groq import GroqProvider
        return GroqProvider()
    elif name == "together":
        from services.llm.together import TogetherProvider
        return TogetherProvider()
    elif name == "openai":
        from services.llm.openai import OpenAIProvider
        return OpenAIProvider()
    else:
        from services.llm.ollama import OllamaProvider
        return OllamaProvider()
