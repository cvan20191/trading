"""
LLM-powered summary engine.

Supports OpenAI and OpenRouter (or any OpenAI-compatible API) via OPENAI_BASE_URL.
Attempts structured output (json_schema) first; if the provider doesn't support it,
retries with json_object mode and manual JSON parsing. Any remaining failure falls
back to the deterministic template in fallback.py so the endpoint always returns
a valid PlaybookSummary.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from openai import AsyncOpenAI, BadRequestError

from app.config import settings
from app.schemas.dashboard_state import DashboardState
from app.schemas.playbook_conclusion import PlaybookConclusion
from app.schemas.summary import (
    PLAYBOOK_SUMMARY_SCHEMA,
    PlaybookSummary,
    PlaybookSummaryContent,
    SummaryMeta,
)
from app.services.fallback import build_fallback_summary
from app.services.prompts import CONCLUSION_INSTRUCTIONS, DEVELOPER_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client  # noqa: PLW0603
    if _client is None:
        kwargs: dict = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
            logger.info("LLM client using custom base URL: %s", settings.openai_base_url)
        _client = AsyncOpenAI(**kwargs)
    return _client


def _build_user_message(
    state: DashboardState,
    conclusion: PlaybookConclusion | None,
    schema_hint: str | None = None,
) -> str:
    sections: list[str] = [DEVELOPER_PROMPT]
    if schema_hint is not None:
        sections.append(
            f"Your response MUST be a JSON object matching this exact schema:\n{schema_hint}"
        )
    if conclusion is not None:
        sections.append(CONCLUSION_INSTRUCTIONS)

    state_json = state.model_dump_json(indent=2)
    sections.append(f"Dashboard state:\n{state_json}")

    if conclusion is not None:
        conclusion_json = conclusion.model_dump_json(indent=2)
        sections.append(f"Playbook conclusion:\n{conclusion_json}")

    return "\n\n".join(sections)


async def generate_summary(
    state: DashboardState,
    conclusion: PlaybookConclusion | None = None,
) -> PlaybookSummary:
    """
    Generate a PlaybookSummary for the given dashboard state.

    Tries structured-output path first, then json_object fallback,
    then deterministic template fallback.
    """
    model = settings.openai_model
    try:
        summary = await _call_llm(state, model, conclusion=conclusion)
        logger.info("LLM summary generated via %s", model)
        return summary
    except Exception as exc:
        logger.warning(
            "LLM summary failed (%s: %s) — using deterministic fallback",
            type(exc).__name__,
            exc,
        )
        return build_fallback_summary(state, model_name=model, conclusion=conclusion)


async def _call_llm(
    state: DashboardState,
    model: str,
    conclusion: PlaybookConclusion | None = None,
) -> PlaybookSummary:
    """
    Route to json_schema or json_object mode based on config.

    json_schema (strict) is only supported by native OpenAI endpoints.
    OpenRouter and other compatible providers should use OPENAI_USE_JSON_SCHEMA=false
    to skip straight to json_object, cutting latency by ~half.
    """
    if not settings.openai_use_json_schema:
        return await _call_with_json_object(state, model, conclusion=conclusion)

    try:
        return await _call_with_json_schema(state, model, conclusion=conclusion)
    except Exception as exc:
        # Provider rejected strict mode or returned wrong shape — retry with json_object
        logger.info(
            "json_schema path failed for %s (%s: %s) — retrying with json_object",
            model, type(exc).__name__, str(exc)[:120],
        )
        return await _call_with_json_object(state, model, conclusion=conclusion)


async def _call_with_json_schema(
    state: DashboardState,
    model: str,
    conclusion: PlaybookConclusion | None = None,
) -> PlaybookSummary:
    client = _get_client()
    user_message = _build_user_message(state, conclusion)

    response = await client.chat.completions.create(
        model=model,
        temperature=settings.openai_temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_message,
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "PlaybookSummary",
                "strict": True,
                "schema": PLAYBOOK_SUMMARY_SCHEMA,
            },
        },
        timeout=settings.openai_timeout,
    )

    return _parse_response(response, model, state)


async def _call_with_json_object(
    state: DashboardState,
    model: str,
    conclusion: PlaybookConclusion | None = None,
) -> PlaybookSummary:
    """
    Fallback path for providers that don't support json_schema strict mode.
    Uses json_object mode and relies on the prompt to produce the right shape.
    """
    client = _get_client()
    schema_hint = json.dumps(PLAYBOOK_SUMMARY_SCHEMA, indent=2)
    augmented_prompt = _build_user_message(
        state,
        conclusion,
        schema_hint=schema_hint,
    )

    response = await client.chat.completions.create(
        model=model,
        temperature=settings.openai_temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_prompt},
        ],
        response_format={"type": "json_object"},
        timeout=settings.openai_timeout,
    )

    return _parse_response(response, model, state)


def _parse_response(response, model: str, state: DashboardState) -> PlaybookSummary:
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("LLM returned empty response content")

    parsed = json.loads(raw)
    content = PlaybookSummaryContent.model_validate(parsed)

    meta = SummaryMeta(
        used_fallback=False,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=model,
        data_status=state.data_freshness.overall_status,
    )

    return PlaybookSummary(**content.model_dump(), meta=meta)
