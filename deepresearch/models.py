"""State models used by the deep research workflow."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import List, Optional

from typing_extensions import Annotated


@dataclass(kw_only=True)
class TodoItem:
    """A single research task."""

    id: int
    title: str
    intent: str
    query: str
    status: str = field(default="pending")
    summary: Optional[str] = field(default=None)
    sources_summary: Optional[str] = field(default=None)
    notices: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class SummaryState:
    """Accumulated state across the research workflow."""

    research_topic: str = field(default=None)
    search_query: str = field(default=None)
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)
    running_summary: str = field(default=None)
    todo_items: Annotated[list, operator.add] = field(default_factory=list)
    structured_report: Optional[str] = field(default=None)


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)
    report_markdown: Optional[str] = field(default=None)
    todo_items: List[TodoItem] = field(default_factory=list)
