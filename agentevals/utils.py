"""LLM utility helpers for agentevals skills."""

from __future__ import annotations

import json
import os
from typing import Any


def _get_openai_client():
    """Return an ``openai.OpenAI`` client, lazily imported."""
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required to use LLM-backed skills. "
            "Install it with: pip install openai"
        ) from exc
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    return openai.OpenAI(api_key=api_key)


def llm_score(
    prompt: str,
    model: str = "gpt-4o-mini",
    *,
    system: str = "You are an impartial evaluator. Follow the instructions exactly.",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call an LLM and parse a JSON response containing ``score`` and ``reasoning``.

    The *prompt* must instruct the model to reply with valid JSON of the form::

        {"score": <float 0-1>, "reasoning": "<explanation>"}

    Returns:
        A ``dict`` with keys ``score`` (float) and ``reasoning`` (str).

    Raises:
        ValueError: If the LLM response cannot be parsed as the expected JSON.
    """
    client = _get_openai_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON content: {raw!r}") from exc

    if "score" not in data:
        raise ValueError(f"LLM response missing 'score' key: {data}")

    return {
        "score": float(data["score"]),
        "reasoning": str(data.get("reasoning", "")),
    }
