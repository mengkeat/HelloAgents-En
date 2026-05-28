"""Search dispatch helpers using the project's existing WebSearch module."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from ..config import Configuration
from ..utils import deduplicate_and_format_sources, format_sources, get_config_value

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()


def _ddgs_search_structured(query: str, max_results: int = 5) -> dict[str, Any]:
    """Run a DuckDuckGo search and return structured results."""
    try:
        from ddgs import DDGS  # noqa: WPS433

        results = DDGS().text(query, max_results=max_results)
        items = []
        for r in (results or []):
            items.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "content": r.get("body", ""),
            })
        return {"results": items, "backend": "duckduckgo", "answer": None, "notices": []}
    except Exception as exc:
        logger.exception("DuckDuckGo search failed: %s", exc)
        return {"results": [], "backend": "duckduckgo", "answer": None, "notices": [str(exc)]}


def _serp_search_structured(query: str, max_results: int = 5) -> dict[str, Any]:
    """Run a SerpApi search and return structured results."""
    try:
        from WebSearch import serp_search  # noqa: WPS433

        raw = serp_search(query)
        if raw.startswith("Error"):
            return {"results": [], "backend": "serpapi", "answer": raw, "notices": []}

        items = []
        # serp_search returns plain text; wrap as a single result
        items.append({
            "title": query,
            "url": "",
            "content": raw,
        })
        return {"results": items, "backend": "serpapi", "answer": raw, "notices": []}
    except Exception as exc:
        logger.exception("SerpApi search failed: %s", exc)
        return {"results": [], "backend": "serpapi", "answer": None, "notices": [str(exc)]}


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """Execute the configured search backend and normalise the response."""

    search_api = get_config_value(config.search_api)
    notices: list[str] = []

    if search_api == "serpapi":
        payload = _serp_search_structured(query)
    else:
        payload = _ddgs_search_structured(query)

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])
    notices = list(payload.get("notices") or [])

    logger.info(
        "Search backend=%s results=%d answer=%s",
        backend_label,
        len(results),
        bool(answer_text),
    )

    return payload, notices, answer_text, backend_label


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_chars_per_source=MAX_TOKENS_PER_SOURCE * 4,
    )

    if answer_text:
        context = f"AI direct answer:\n{answer_text}\n\n{context}"

    return sources_summary, context
