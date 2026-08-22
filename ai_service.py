"""OpenRouter integration for the contextual CognitiveNeuro assistant."""

import json
import os
from typing import Any

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/free"
MISSING_CREDENTIALS_MESSAGE = "AI assistant unavailable — API credentials have not been configured."
GENERIC_ERROR_MESSAGE = "The CognitiveNeuro Assistant is temporarily unavailable. Please try again shortly."
RATE_LIMIT_MESSAGE = (
    "The CognitiveNeuro Assistant is temporarily busy due to a service rate limit. "
    "Please try again shortly."
)

HTTP_SESSION = requests.Session()
HTTP_SESSION.trust_env = False

SYSTEM_PROMPT = """You are the CognitiveNeuro educational assistant embedded in an
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
When a current analysis is supplied and the user asks to analyze the page or asks
what to do next, use those exact facts: briefly summarize the result, explain the
most relevant EEG-band and recording information, identify uncertainty or quality
considerations, and give 3-5 practical, non-diagnostic next steps tailored to the
analysis. Treat references such as "this," "this page," "my result," "the upload,"
"analyze this," "explain this," and "next steps" as references to the supplied
current analysis unless the user clearly indicates otherwise. Do not ask the user
to repeat facts already present in the context. Do not give generic advice when
application-provided facts are available. If no analysis is supplied, say that a
completed EEG analysis is needed and never invent results."""


def _analysis_summary(context: dict[str, Any] | None) -> str:
    """Create a bounded summary; raw CSV samples are never included."""
    if not context:
        return "No EEG analysis is currently available."

    result = context.get("result", {})
    bands = context.get("bands", {})
    recording = context.get("recording") or {}
    feature_values = context.get("feature_values", {})

    band_text = ", ".join(
        f"{name}={float(value):.5g}"
        for name, value in bands.items()
        if isinstance(value, (int, float))
    ) or "unavailable"

    feature_groups = {}
    for suffix in ("std", "mean", "theta_alpha_ratio", "slowing_index"):
        values = [
            float(value)
            for name, value in feature_values.items()
            if name.endswith(f"_{suffix}") and isinstance(value, (int, float))
        ]
        if values:
            feature_groups[suffix] = (
                f"mean={sum(values) / len(values):.5g}, "
                f"range={min(values):.5g}..{max(values):.5g}"
            )
    feature_text = "; ".join(
        f"{name}: {summary}" for name, summary in feature_groups.items()
    ) or "aggregate feature detail unavailable"

    probability = result.get("pattern_probability")
    comparison = result.get("control_probability")
    probability_text = f"{float(probability) * 100:.1f}%" if isinstance(probability, (int, float)) else "unavailable"
    comparison_text = f"{float(comparison) * 100:.1f}%" if isinstance(comparison, (int, float)) else "unavailable"
    lines = [
        "CURRENT PAGE AND EEG ANALYSIS CONTEXT (trusted application-provided facts):",
        f"- Current page: {context.get('page', 'EEG Analysis')}",
        f"- Input source: {context.get('source', 'unknown')}",
        f"- Analysis timestamp: {context.get('timestamp', 'unknown')}",
        f"- Displayed EEG band values: {band_text}",
        f"- Compact 64-feature summary across channels: {feature_text}",
        f"- Random Forest AD-pattern similarity: {probability_text}",
        f"- Random Forest comparison-class probability: {comparison_text}",
        f"- Records analyzed: {result.get('records', 'unavailable')}",
        f"- AD-pattern records: {result.get('pattern_records', 'unavailable')}",
        f"- Experimental Random Forest verdict: {context.get('verdict', 'unavailable')}",
        "- Model: RandomForestClassifier; input validated; not clinically validated",
    ]
    if recording:
        lines.append(
            "- Validated recording: "
            f"{recording.get('samples', 'unknown')} samples, "
            f"{recording.get('duration_seconds', 'unknown')} seconds, "
            f"{recording.get('complete_epochs', 'unknown')} complete epochs, "
            f"{recording.get('channels', 'unknown')} channels"
        )
    lines.extend((
        "- No raw CSV rows or individual EEG samples are included in this context.",
        "Use this context when relevant. Do not reinterpret it as a diagnosis or clinical probability.",
    ))
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
