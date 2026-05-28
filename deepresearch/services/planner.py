"""Service responsible for converting the research topic into actionable tasks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from ..config import Configuration
from ..llm_service import HelloAgentsLLM, get_llm
from ..models import SummaryState, TodoItem
from ..prompts import get_current_date, todo_planner_instructions
from ..utils import strip_thinking_tokens

logger = logging.getLogger(__name__)


class PlanningService:
    """Uses the shared LLM to decompose a topic into structured TODO items."""

    def __init__(self, llm_client: HelloAgentsLLM | None = None, config: Configuration | None = None) -> None:
        self._llm = llm_client
        self._config = config or Configuration.from_env()

    @property
    def llm(self) -> HelloAgentsLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def plan_todo_list(self, state: SummaryState) -> List[TodoItem]:
        """Ask the planner to break the topic into actionable tasks."""

        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )

        response = self.llm.think_simple(prompt) or ""
        logger.info("Planner raw output (truncated): %s", response[:500])

        tasks_payload = self._extract_tasks(response)
        todo_items: List[TodoItem] = []

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"Task {idx}").strip()
            intent = str(item.get("intent") or "Focus on a key aspect of the topic").strip()
            query = str(item.get("query") or state.research_topic).strip()

            if not query:
                query = state.research_topic

            todo_items.append(TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
            ))

        state.todo_items = todo_items
        logger.info("Planner produced %d tasks: %s", len(todo_items), [t.title for t in todo_items])
        return todo_items

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """Create a minimal fallback task when planning fails."""
        return TodoItem(
            id=1,
            title="Background research",
            intent="Gather core background and latest developments on the topic",
            query=f"{state.research_topic} latest developments" if state.research_topic else "background research",
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_response: str) -> List[dict[str, Any]]:
        """Parse planner output into a list of task dictionaries."""
        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        json_payload = self._extract_json_payload(text)
        tasks: List[dict[str, Any]] = []

        if isinstance(json_payload, dict):
            candidate = json_payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)

        return tasks

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[dict[str, Any] | list]:
        """Try to locate and parse a JSON object or array from the text."""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None
