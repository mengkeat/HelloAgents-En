# Deep Research Agent

English local port of Datawhale's chapter 14 deep-research agent example.

This version is Python-only and uses this repository's existing
`src/HelloAgentsLLM.py` LiteLLM wrapper instead of the upstream
`hello-agents` package. Runtime and dependencies are managed by `uv`.

## How it works

1. **Plan** — the LLM decomposes the research topic into 3–5 focused tasks.
2. **Search** — each task triggers a web search (DuckDuckGo by default).
3. **Summarise** — the LLM produces a detailed summary per task from the search results.
4. **Report** — the LLM consolidates all task summaries into a structured Markdown report.

## Environment

Create a `.env` file in the repository root if you want live LLM and search data:

```env
LLM_MODEL_ID=deepseek/deepseek-v4-pro
DEEPSEEK_API_KEY=your-deepseek-api-key
LLM_TIMEOUT=60

SEARCH_API=duckduckgo
# SERPAPI_API_KEY=your-serpapi-key   # required if SEARCH_API=serpapi

MAX_WEB_RESEARCH_LOOPS=3
```

## Run

```bash
uv run python -m deepresearch "What are the latest advances in quantum computing?"
```

Write the report to a file:

```bash
uv run python -m deepresearch "Transformer architecture evolution" --output report.md
```

Output full JSON with task metadata:

```bash
uv run python -m deepresearch "Rust vs Go for backend services" --json --output result.json
```

Use SerpApi instead of DuckDuckGo:

```bash
uv run python -m deepresearch "History of the internet" --search serpapi
```

## Use from Python

```python
from deepresearch import run_deep_research

result = run_deep_research("What is retrieval-augmented generation?")
print(result.report_markdown)
for task in result.todo_items:
    print(f"  [{task.status}] {task.title}: {task.summary[:80]}...")
```
