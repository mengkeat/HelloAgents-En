"""Task summarization service using the shared LLM."""

from __future__ import annotations

from typing import Optional

from ..config import Configuration
from ..llm_service import HelloAgentsLLM, get_llm
from ..models import SummaryState, TodoItem
from ..utils import strip_thinking_tokens
from .text_processing import strip_tool_calls


class SummarizationService:
    """Generates task-level summaries via the shared LLM."""

    def __init__(self, llm_client: HelloAgentsLLM | None = None, config: Configuration | None = None) -> None:
        self._llm = llm_client
        self._config = config or Configuration.from_env()

    @property
    def llm(self) -> HelloAgentsLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def summarize_task(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """Generate a task-specific summary."""

        prompt = self._build_prompt(state, task, context)
        response = self.llm.think_simple(prompt) or ""

        summary_text = response.strip()
        if self._config.strip_thinking_tokens:
            summary_text = strip_thinking_tokens(summary_text)

        summary_text = strip_tool_calls(summary_text).strip()
        return summary_text or "No information available."

    @staticmethod
    def _build_prompt(state: SummaryState, task: TodoItem, context: str) -> str:
        from ..prompts import task_summarizer_instructions

        return (
            f"{task_summarizer_instructions.strip()}\n\n"
            f"Research topic: {state.research_topic}\n"
            f"Task name: {task.title}\n"
            f"Task goal: {task.intent}\n"
            f"Search query: {task.query}\n"
            f"Task context:\n{context}\n\n"
            "Please provide a thorough Markdown summary following the task summary template above."
        )
