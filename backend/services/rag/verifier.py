"""Self-RAG verification — checks that generated field values are grounded in sources.

Asks the LLM to cite the supporting passage for each generated field.
Returns a confidence score (0–1) stored as confidence_scores JSONB in the flowers table.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from services.llm.provider import LLMProvider


class VerificationResult(BaseModel):
    supported: bool = False
    quote: str = "none"
    confidence: float = 0.0


async def verify_field(
    field_name: str,
    field_value: str,
    source_text: str,
    llm: LLMProvider,
) -> VerificationResult:
    """Verify that a generated field value is supported by the source material."""
    if not field_value or field_value == "Information not available.":
        return VerificationResult(supported=False, quote="none", confidence=0.0)

    response = await llm.complete(
        prompt=f"""Rate how well this claim is supported by the source material.

Claim: {field_name} = "{field_value[:300]}"

Source material:
{source_text[:5000]}

Reply with a single decimal number from 0.0 to 1.0:
- 1.0 = claim is explicitly stated in the source
- 0.5 = claim is partially supported or implied
- 0.0 = claim is not found in the source

Reply with ONLY the number, nothing else.""",
        system="You are a fact-checking assistant. Reply with only a decimal number.",
    )

    return _parse_verification(response)


async def verify_all_fields(
    generated_fields: dict[str, str],
    source_text: str,
    llm: LLMProvider,
) -> dict[str, VerificationResult]:
    """Verify all generated text fields against combined source text."""
    results: dict[str, VerificationResult] = {}
    for field_name, field_value in generated_fields.items():
        result = await verify_field(field_name, field_value, source_text, llm)
        results[field_name] = result
    return results


def _parse_verification(response: str) -> VerificationResult:
    text = response.strip()
    # Extract first float-like token from the response
    match = re.search(r"\b(0?\.\d+|1\.0|[01])\b", text)
    if match:
        try:
            score = max(0.0, min(1.0, float(match.group())))
            return VerificationResult(supported=score >= 0.5, confidence=score)
        except ValueError:
            pass
    return VerificationResult()
