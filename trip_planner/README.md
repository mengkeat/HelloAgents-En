# Trip Planner Agent

English local port of Datawhale's chapter 13 trip-planner agent example.

This version is intentionally Python-only and uses this repository's existing
`src/HelloAgentsLLM.py` LiteLLM wrapper instead of adding the upstream
`hello-agents` package. Runtime and dependencies are managed by `uv`.

## Environment

Create a `.env` file in the repository root if you want live LLM and map data:

```env
LLM_MODEL_ID=deepseek/deepseek-v4-pro
DEEPSEEK_API_KEY=your-deepseek-api-key
# LLM_BASE_URL is optional for OpenAI-compatible providers or LiteLLM proxy usage
LLM_TIMEOUT=60

AMAP_API_KEY=your-amap-web-service-key
# Optional: use src/WebSearch.py when map attraction data is unavailable
TRIP_PLANNER_USE_WEB_SEARCH=0
```

If the LLM or Amap key is missing, the agent still returns a deterministic
fallback itinerary so the CLI can run locally.

## Run

```bash
uv run python -m trip_planner \
  --city Beijing \
  --start-date 2026-06-01 \
  --end-date 2026-06-03 \
  --transportation "public transit" \
  --accommodation "mid-range hotel" \
  --preference history \
  --preference "local food"
```

Write JSON to a file:

```bash
uv run python -m trip_planner --city Beijing --start-date 2026-06-01 --end-date 2026-06-03 --output trip.json
```

## Use from Python

```python
from trip_planner import TripRequest, get_trip_planner_agent

request = TripRequest(
    city="Beijing",
    start_date="2026-06-01",
    end_date="2026-06-03",
    preferences=["history", "local food"],
)

plan = get_trip_planner_agent().plan_trip(request)
print(plan.model_dump_json(indent=2))
```
