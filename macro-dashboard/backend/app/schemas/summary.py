from __future__ import annotations

from pydantic import BaseModel, model_validator

# Filler strings used when the LLM or state does not supply enough items
FILLER_WATCH = "No additional watchpoint available"
FILLER_CHANGED = "No major change detected"
FILLER_TRIGGER = "No additional trigger provided"


def pad_or_truncate(items: list[str], length: int, filler: str) -> list[str]:
    """Return exactly `length` items, truncating extras or padding with `filler`."""
    result = list(items)[:length]
    while len(result) < length:
        result.append(filler)
    return result


class SummaryMeta(BaseModel):
    used_fallback: bool
    generated_at: str
    model: str | None = None
    data_status: str = "unknown"


class PlaybookSummaryContent(BaseModel):
    """The 9 content fields that the LLM generates (no meta)."""

    headline_summary: str
    expanded_summary: str
    regime_label: str
    posture_label: str
    watch_now: list[str]
    what_changed_bullets: list[str]
    what_changes_call_bullets: list[str]
    risk_flags: list[str]
    teaching_note: str

    @model_validator(mode="after")
    def enforce_array_lengths(self) -> PlaybookSummaryContent:
        self.watch_now = pad_or_truncate(self.watch_now, 3, FILLER_WATCH)
        self.what_changed_bullets = pad_or_truncate(
            self.what_changed_bullets, 3, FILLER_CHANGED
        )
        self.what_changes_call_bullets = pad_or_truncate(
            self.what_changes_call_bullets, 3, FILLER_TRIGGER
        )
        return self


class PlaybookSummary(PlaybookSummaryContent):
    """Full API response: content fields plus server-side metadata."""

    meta: SummaryMeta


# ---------------------------------------------------------------------------
# JSON Schema for OpenAI structured output
# Only covers the 9 content fields the model should produce (no meta).
# strict=True requires additionalProperties=false and all fields in required.
# ---------------------------------------------------------------------------
PLAYBOOK_SUMMARY_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "headline_summary",
        "expanded_summary",
        "regime_label",
        "posture_label",
        "watch_now",
        "what_changed_bullets",
        "what_changes_call_bullets",
        "risk_flags",
        "teaching_note",
    ],
    "properties": {
        "headline_summary": {
            "type": "string",
            "description": "Exactly 2 sentences summarising current regime and implied posture.",
        },
        "expanded_summary": {
            "type": "string",
            "description": "One paragraph of 3–5 sentences explaining why the regime exists.",
        },
        "regime_label": {
            "type": "string",
            "description": "Primary regime name from the dashboard engine.",
        },
        "posture_label": {
            "type": "string",
            "description": "Short posture phrase such as 'Hold and wait'.",
        },
        "watch_now": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exactly 3 watchpoint bullets.",
        },
        "what_changed_bullets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exactly 3 bullets describing recent changes.",
        },
        "what_changes_call_bullets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exactly 3 bullets describing what would flip the regime call.",
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short tag labels for active risk overlays.",
        },
        "teaching_note": {
            "type": "string",
            "description": "One educational sentence explaining a key framework mechanic.",
        },
    },
}
