"""OpenRouter integration for the contextual NeuroSage assistant."""

import json
import os
from typing import Any

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/free"
MISSING_CREDENTIALS_MESSAGE = "AI assistant unavailable — API credentials have not been configured."
GENERIC_ERROR_MESSAGE = "The NeuroSage Assistant is temporarily unavailable. Please try again shortly."
RATE_LIMIT_MESSAGE = (
    "The NeuroSage Assistant is temporarily busy due to a service rate limit. "
    "Please try again shortly."
)

HTTP_SESSION = requests.Session()
HTTP_SESSION.trust_env = False

SYSTEM_PROMPT = """You are the NeuroSage educational assistant embedded in an
experimental EEG research application. Explain the supplied analysis in plain,
careful language. The Random Forest is solely responsible for every numerical
prediction; never calculate, alter, or invent a score. Never diagnose Alzheimer's
disease, estimate a person's clinical disease probability, prescribe medication,
or recommend starting, stopping, or changing medication. Never describe this
experimental model as clinically validated. Clearly distinguish Alzheimer's-pattern
similarity from diagnosis or clinical risk. You may explain EEG features, uncertainty,
research limitations, recording-quality considerations, and cautious questions or
follow-up topics to discuss with a qualified clinician. If asked for disallowed
medical advice, state the limitation and redirect to appropriate professional care.
Keep each answer concise and complete: answer the question directly in no more than
three short sentences and about 90 words. Avoid introductions, repetition, and
unrequested background detail. Never end mid-sentence; preserve essential safety
context even when brevity requires omitting secondary detail."""


def _analysis_summary(context: dict[str, Any] | None) -> str:
    """Create a bounded summary; raw CSV samples are never included."""
    if not context:
        return "No EEG analysis is currently available."

    result = context.get("result", {})
    lines = [
        "Current experimental EEG analysis (application-provided facts):",
        f"- Input source: {context.get('source', 'unknown')}",
        f"- Analysis timestamp: {context.get('timestamp', 'unknown')}",
        f"- Displayed EEG band values: {context.get('bands', {})}",
        f"- Model feature values (compact aggregate, not raw CSV): {context.get('feature_values', {})}",
        f"- Random Forest AD-pattern similarity: {result.get('pattern_probability', 'unavailable')}",
        f"- Random Forest comparison-class probability: {result.get('control_probability', 'unavailable')}",
        f"- Records analyzed: {result.get('records', 'unavailable')}",
        f"- AD-pattern records: {result.get('pattern_records', 'unavailable')}",
        f"- Experimental Random Forest verdict: {context.get('verdict', 'unavailable')}",
        f"- Model/validation metadata: {context.get('metadata', {})}",
    ]
    if context.get("recording"):
        lines.append(f"- Validated recording metadata: {context['recording']}")
    lines.append("Do not reinterpret these values as a diagnosis or clinical probability.")
    return "\n".join(lines)


def answer_question(
    question: str,
    context: dict[str, Any] | None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Answer through OpenRouter, returning only safe user-facing errors."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return MISSING_CREDENTIALS_MESSAGE

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _analysis_summary(context)},
    ]
    for message in (history or [])[-6:]:
        role, content = message.get("role"), message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content[:2000]})
    messages.append({"role": "user", "content": question[:2000]})

    try:
        response = HTTP_SESSION.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "max_tokens": 450,
                "temperature": 0.2,
            },
            timeout=(4, 18),
        )
        if response.status_code == 429:
            return RATE_LIMIT_MESSAGE
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip() if isinstance(content, str) and content.strip() else GENERIC_ERROR_MESSAGE
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return GENERIC_ERROR_MESSAGE


def answer_question_stream(
    question: str,
    context: dict[str, Any] | None,
    history: list[dict[str, str]] | None = None,
):
    """Yield an OpenRouter response as it arrives for immediate UI feedback."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        yield MISSING_CREDENTIALS_MESSAGE
        return

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _analysis_summary(context)},
    ]
    for message in (history or [])[-6:]:
        role, content = message.get("role"), message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content[:2000]})
    messages.append({"role": "user", "content": question[:2000]})

    try:
        with HTTP_SESSION.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "max_tokens": 450,
                "temperature": 0.2,
                "stream": True,
            },
            timeout=(4, 18),
            stream=True,
        ) as response:
            if response.status_code == 429:
                yield RATE_LIMIT_MESSAGE
                return
            response.raise_for_status()
            response.encoding = "utf-8"
            received_content = False
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                    content = event["choices"][0].get("delta", {}).get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if content:
                    received_content = True
                    yield content
            if not received_content:
                yield GENERIC_ERROR_MESSAGE
    except requests.RequestException:
        yield GENERIC_ERROR_MESSAGE
