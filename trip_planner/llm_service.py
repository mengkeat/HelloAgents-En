"""LLM service adapter that reuses the existing src/HelloAgentsLLM client."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from HelloAgentsLLM import HelloAgentsLLM  # noqa: E402

_llm_instance: Optional[HelloAgentsLLM] = None


def get_llm() -> HelloAgentsLLM:
    """Return a singleton HelloAgentsLLM instance using the project's existing client."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = HelloAgentsLLM()
    return _llm_instance


def reset_llm() -> None:
    """Reset the cached LLM instance."""
    global _llm_instance
    _llm_instance = None
