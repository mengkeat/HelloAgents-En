"""Configuration for the deep research agent."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchAPI(Enum):
    DUCKDUCKGO = "duckduckgo"
    SERPAPI = "serpapi"


class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        description="Number of research iterations to perform",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        description="Web search API to use",
    )
    fetch_full_page: bool = Field(
        default=False,
        description="Include the full page content in the search results",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        description="Whether to strip thinking tokens from model responses",
    )

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "Configuration":
        """Create a configuration from environment variables and overrides."""
        raw: dict[str, Any] = {}

        for field_name in cls.model_fields:
            env_key = field_name.upper()
            if env_key in os.environ:
                raw[field_name] = os.environ[env_key]

        # Explicit aliases
        aliases = {
            "max_web_research_loops": os.getenv("MAX_WEB_RESEARCH_LOOPS"),
            "fetch_full_page": os.getenv("FETCH_FULL_PAGE"),
            "strip_thinking_tokens": os.getenv("STRIP_THINKING_TOKENS"),
            "search_api": os.getenv("SEARCH_API"),
        }
        for key, value in aliases.items():
            if value is not None:
                raw.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw[key] = value

        return cls(**raw)
