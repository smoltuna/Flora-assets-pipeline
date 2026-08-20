import asyncio

from anthropic import (
    NOT_GIVEN,
    APIConnectionError,
    APIStatusError,
    AsyncAnthropicBedrock,
)
from config import settings

from services.llm.provider import LLMResponse

_DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_MAX_TOKENS = 4096
_MAX_ATTEMPTS = 4
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class BedrockProvider:
    def __init__(self, model_override: str | None = None) -> None:
        region = settings.aws_region or "us-east-1"
        api_key = settings.aws_bearer_token_bedrock or None
        self.client = AsyncAnthropicBedrock(aws_region=region, api_key=api_key)
        self.model = model_override or settings.bedrock_model or _DEFAULT_MODEL

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        backoff = 5.0
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=_MAX_TOKENS,
                    system=system if system else NOT_GIVEN,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
                text = "".join(text_parts)
                if not text.strip():
                    raise RuntimeError("Bedrock returned empty completion content")

                tokens = message.usage.input_tokens + message.usage.output_tokens
                from services.llm import _token_counter
                _token_counter.record(tokens)
                return LLMResponse(text=text, tokens_used=tokens)

            except APIStatusError as exc:
                if exc.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise
            except APIConnectionError:
                if attempt >= _MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2

        raise RuntimeError("Bedrock: max retries exceeded")
