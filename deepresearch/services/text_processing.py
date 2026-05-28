"""Utility helpers for normalizing agent-generated text."""

from __future__ import annotations

import re


def strip_tool_calls(text: str) -> str:
    """Remove tool-call markers from text."""

    if not text:
        return text

    pattern = re.compile(r"\[TOOL_CALL:[^\]]+\]")
    return pattern.sub("", text)
