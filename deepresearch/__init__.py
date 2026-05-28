"""Deep Research agent package.

A deep research assistant that decomposes a topic into tasks, searches the
web, summarises findings, and produces a structured Markdown report.
"""

from .agent import DeepResearchAgent, run_deep_research
from .config import Configuration, SearchAPI
from .models import SummaryState, SummaryStateOutput, TodoItem

__all__ = [
    "DeepResearchAgent",
    "Configuration",
    "SearchAPI",
    "SummaryState",
    "SummaryStateOutput",
    "TodoItem",
    "run_deep_research",
]
