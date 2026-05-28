"""Orchestrator coordinating the deep research workflow.

This implementation follows the multi-stage pattern of the upstream example:
1. Plan TODO tasks from the research topic.
2. For each task: search the web → summarise findings.
3. Consolidate all task summaries into a structured Markdown report.

It reuses the repository's existing ``HelloAgentsLLM`` wrapper instead of the
upstream ``hello-agents`` package, matching the pattern established by
``trip_planner``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import Configuration
from .llm_service import HelloAgentsLLM, get_llm
from .models import SummaryState, SummaryStateOutput, TodoItem
from .services.planner import PlanningService
from .services.reporter import ReportingService
from .services.search import dispatch_search, prepare_research_context
from .services.summarizer import SummarizationService

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """Coordinator that orchestrates TODO-based research using the shared LLM."""

    def __init__(
        self,
        llm_client: HelloAgentsLLM | None = None,
        config: Configuration | None = None,
    ) -> None:
        self._llm_client = llm_client
        self.config = config or Configuration.from_env()
        self.planner = PlanningService(self.llm_client, self.config)
        self.summarizer = SummarizationService(self.llm_client, self.config)
        self.reporting = ReportingService(self.llm_client, self.config)

    @property
    def llm_client(self) -> HelloAgentsLLM:
        if self._llm_client is None:
            self._llm_client = get_llm()
        return self._llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, topic: str) -> SummaryStateOutput:
        """Execute the full research workflow and return the final report."""
        print(f"\n{'='*60}")
        print(f"  Deep Research: {topic}")
        print(f"{'='*60}\n")

        state = SummaryState(research_topic=topic)

        # Step 1 — Plan tasks
        print("📋 Planning research tasks...")
        state.todo_items = self.planner.plan_todo_list(state)
        if not state.todo_items:
            logger.info("No tasks generated; falling back to single task.")
            state.todo_items = [PlanningService.create_fallback_task(state)]

        print(f"   Generated {len(state.todo_items)} research task(s):")
        for t in state.todo_items:
            print(f"   • [{t.id}] {t.title} — {t.intent}")

        # Step 2 — Execute tasks sequentially
        for task in state.todo_items:
            self._execute_task(state, task)

        # Step 3 — Generate report
        print("\n📝 Generating final report...")
        report = self.reporting.generate_report(state)
        state.structured_report = report
        state.running_summary = report

        print(f"\n{'='*60}")
        print("  Research complete")
        print(f"{'='*60}\n")

        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=state.todo_items,
        )

    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        """Execute the workflow yielding incremental progress events.

        This is designed for programmatic consumers that want to display
        progress in real time (e.g. a TUI or web frontend).
        """
        state = SummaryState(research_topic=topic)
        yield {"type": "status", "message": "Initializing research workflow"}

        # Step 1 — Plan
        print("📋 Planning research tasks...")
        state.todo_items = self.planner.plan_todo_list(state)
        if not state.todo_items:
            state.todo_items = [PlanningService.create_fallback_task(state)]

        yield {
            "type": "todo_list",
            "tasks": [self._serialize_task(t) for t in state.todo_items],
        }

        # Step 2 — Execute tasks
        for task in state.todo_items:
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "in_progress",
                "title": task.title,
                "intent": task.intent,
            }

            for event in self._execute_task_stream(state, task):
                yield event

            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": task.status,
                "summary": task.summary,
                "sources_summary": task.sources_summary,
            }

        # Step 3 — Report
        print("\n📝 Generating final report...")
        report = self.reporting.generate_report(state)
        state.structured_report = report
        state.running_summary = report

        yield {"type": "final_report", "report": report}
        yield {"type": "done"}

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_task(self, state: SummaryState, task: TodoItem) -> None:
        """Run search + summarization for a single task (non-streaming)."""
        task.status = "in_progress"
        print(f"\n🔍 Task {task.id}: {task.title}")
        print(f"   Query: {task.query}")

        search_result, notices, answer_text, backend = dispatch_search(
            task.query, self.config, state.research_loop_count,
        )
        task.notices = notices

        if not search_result or not search_result.get("results"):
            print(f"   ⚠ No results found for this task.")
            task.status = "skipped"
            return

        sources_summary, context = prepare_research_context(
            search_result, answer_text, self.config,
        )
        task.sources_summary = sources_summary

        state.web_research_results.append(context)
        state.sources_gathered.append(sources_summary)
        state.research_loop_count += 1

        print(f"   📊 Found {len(search_result.get('results', []))} source(s), summarising...")
        summary_text = self.summarizer.summarize_task(state, task, context)
        task.summary = summary_text.strip() if summary_text else "No information available."
        task.status = "completed"
        print(f"   ✅ Task {task.id} complete.")

    def _execute_task_stream(self, state: SummaryState, task: TodoItem) -> Iterator[dict[str, Any]]:
        """Run search + summarization for a single task, yielding progress events."""
        task.status = "in_progress"

        search_result, notices, answer_text, backend = dispatch_search(
            task.query, self.config, state.research_loop_count,
        )
        task.notices = notices

        if not search_result or not search_result.get("results"):
            task.status = "skipped"
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "skipped",
            }
            return

        sources_summary, context = prepare_research_context(
            search_result, answer_text, self.config,
        )
        task.sources_summary = sources_summary

        state.web_research_results.append(context)
        state.sources_gathered.append(sources_summary)
        state.research_loop_count += 1

        yield {
            "type": "sources",
            "task_id": task.id,
            "latest_sources": sources_summary,
            "backend": backend,
        }

        summary_text = self.summarizer.summarize_task(state, task, context)
        task.summary = summary_text.strip() if summary_text else "No information available."
        task.status = "completed"

        yield {
            "type": "task_summary_chunk",
            "task_id": task.id,
            "content": task.summary,
        }

    @staticmethod
    def _serialize_task(task: TodoItem) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "status": task.status,
            "summary": task.summary,
            "sources_summary": task.sources_summary,
        }


def run_deep_research(
    topic: str,
    config: Configuration | None = None,
    llm_client: HelloAgentsLLM | None = None,
) -> SummaryStateOutput:
    """Convenience function wrapping :class:`DeepResearchAgent`."""
    agent = DeepResearchAgent(llm_client=llm_client, config=config)
    return agent.run(topic)
