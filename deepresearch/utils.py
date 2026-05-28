"""Utility helpers shared across deep researcher services."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def get_config_value(value: Any) -> str:
    """Return configuration value as plain string."""
    return value if isinstance(value, str) else value.value


def strip_thinking_tokens(text: str) -> str:
    """Remove thinking-token sections from model responses."""
    while "iminal>" in text and "</iminal>" in text:
        start = text.find("iminal>")
        end = text.find("</iminal>") + len("</iminal>")
        text = text[:start] + text[end:]
    # Also handle common thinking tags
    while "<think" in text and "</think" in text:
        start = text.find("<think")
        end = text.find("</think") + len("</think")
        if end <= start:
            break
        text = text[:start] + text[end:]
    return text


def deduplicate_and_format_sources(
    search_response: Dict[str, Any] | List[Dict[str, Any]],
    max_chars_per_source: int = 8000,
) -> str:
    """Format and deduplicate search results for downstream prompting."""

    if isinstance(search_response, dict):
        sources_list = search_response.get("results", [])
    else:
        sources_list = search_response

    unique_sources: dict[str, Dict[str, Any]] = {}
    for source in sources_list:
        url = source.get("url") or source.get("href")
        if not url:
            continue
        if url not in unique_sources:
            unique_sources[url] = source

    formatted_parts: List[str] = []
    for source in unique_sources.values():
        title = source.get("title") or source.get("url", "")
        content = source.get("content") or source.get("body") or ""
        url = source.get("url") or source.get("href") or ""
        formatted_parts.append(f"Source: {title}\n")
        formatted_parts.append(f"URL: {url}\n")
        formatted_parts.append(f"Content: {content}\n\n")

        if len(content) > max_chars_per_source:
            content = f"{content[:max_chars_per_source]}... [truncated]"

    return "".join(formatted_parts).strip()


def format_sources(search_results: Dict[str, Any] | None) -> str:
    """Return bullet list summarising search sources."""

    if not search_results:
        return ""

    results = search_results.get("results", [])
    lines: list[str] = []
    for item in results:
        url = item.get("url") or item.get("href") or ""
        title = item.get("title", url)
        if url:
            lines.append(f"* {title} : {url}")
    return "\n".join(lines)
