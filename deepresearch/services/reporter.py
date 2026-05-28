"""Service that consolidates task results into the final report."""

from __future__ import annotations

from typing import Optional

from ..config import Configuration
from ..llm_service import HelloAgentsLLM, get_llm
from ..models import SummaryState
from ..utils import strip_thinking_tokens
from .text_processing import strip_tool_calls


class ReportingService:
    """Generates the final structured report."""

    def __init__(self, llm_client: HelloAgentsLLM | None = None, config: Configuration | None = None) -> None:
        self._llm = llm_client
        self._config = config or Configuration.from_env()

    @property
    def llm(self) -> HelloAgentsLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def generate_report(self, state: SummaryState) -> str:
        """Generate a structured report based on completed tasks."""

        from ..prompts import report_writer_instructions

        tasks_block: list[str] = []
        for task in state.todo_items:
            summary_block = task.summary or "No information available."
            sources_block = task.sources_summary or "No sources available."
            tasks_block.append(
                f"### Task {task.id}: {task.title}\n"
                f"- Goal: {task.intent}\n"
                f"- Search query: {task.query}\n"
                f"- Status: {task.status}\n"
                f"- Summary:\n{summary_block}\n"
                f"- Sources:\n{sources_block}\n"
            )

        prompt = (
            f"{report_writer_instructions.strip()}\n\n"
            f"Research topic: {state.research_topic}\n\n"
            f"Task overview:\n{''.join(tasks_block)}\n"
            "Please generate the research report following the template above."
        )

        response = self.llm.think_simple(prompt) or ""

        report_text = response.strip()
        if self._config.strip_thinking_tokens:
            report_text = strip_thinking_tokens(report_text)

        report_text = strip_tool_calls(report_text).strip()
        return report_text or "Report generation failed. Please check the input."
