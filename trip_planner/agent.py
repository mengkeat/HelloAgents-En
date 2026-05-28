"""English trip planner agent ported from the chapter 13 example.

This implementation keeps the multi-stage shape of the upstream example:
1. collect attraction context,
2. collect weather context,
3. collect hotel context,
4. ask an LLM to synthesize a complete itinerary.

It reuses the repository's existing `HelloAgentsLLM` wrapper instead of adding the
upstream `hello-agents` dependency.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .amap_service import AmapService
from .llm_service import HelloAgentsLLM, get_llm
from .models import (
    Attraction,
    Budget,
    DayPlan,
    Hotel,
    Location,
    Meal,
    POIInfo,
    TripPlan,
    TripRequest,
    WeatherInfo,
)

PLANNER_PROMPT_TEMPLATE = """
You are a professional travel-planning agent.

Create a practical, day-by-day travel plan from the user request and the tool
context below. Return ONLY a JSON object. Do not include Markdown fences,
commentary, or trailing text.

Important output rules:
- All human-readable text must be in English.
- Keep `days` length equal to `travel_days`; `day_index` starts at 0.
- Each day should contain 2-3 attractions and breakfast, lunch, and dinner.
- Prefer attractions and hotels from the provided map context. Use their exact
  coordinates when available.
- If context is incomplete, make reasonable recommendations and mention the
  uncertainty in `overall_suggestions`.
- Temperature fields must be numbers, not strings with units.
- Budget numbers are estimates in the destination's local currency.

Required JSON shape:
{{
  "city": "City name",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "Day summary",
      "transportation": "Transportation advice",
      "accommodation": "Accommodation advice",
      "hotel": {{
        "name": "Hotel name",
        "address": "Hotel address",
        "location": {{"longitude": 0.0, "latitude": 0.0}},
        "price_range": "Estimated range",
        "rating": "Rating when known",
        "distance": "Useful distance note",
        "type": "Hotel type",
        "estimated_cost": 0
      }},
      "attractions": [
        {{
          "name": "Attraction name",
          "address": "Address",
          "location": {{"longitude": 0.0, "latitude": 0.0}},
          "visit_duration": 120,
          "description": "Why visit",
          "category": "Category",
          "ticket_price": 0
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "Breakfast", "description": "Details", "estimated_cost": 0}},
        {{"type": "lunch", "name": "Lunch", "description": "Details", "estimated_cost": 0}},
        {{"type": "dinner", "name": "Dinner", "description": "Details", "estimated_cost": 0}}
      ]
    }}
  ],
  "weather_info": [
    {{
      "date": "YYYY-MM-DD",
      "day_weather": "Sunny",
      "night_weather": "Cloudy",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "South",
      "wind_power": "1-3"
    }}
  ],
  "overall_suggestions": "General practical advice",
  "budget": {{
    "total_attractions": 0,
    "total_hotels": 0,
    "total_meals": 0,
    "total_transportation": 0,
    "total": 0
  }}
}}

User request:
{request_json}

Map attraction context:
{attraction_context}

Weather context:
{weather_context}

Hotel context:
{hotel_context}

Optional web context:
{web_context}
""".strip()

JSON_REPAIR_PROMPT_TEMPLATE = """
Fix the following travel-plan response so it is valid JSON matching the requested
trip plan schema. Return ONLY the corrected JSON object.

Rules:
- Preserve the intended English content when possible.
- Fix malformed field names such as visit_90 into "visit_duration": 90.
- Use these exact field names: city, start_date, end_date, days, weather_info,
  overall_suggestions, budget, date, day_index, description, transportation,
  accommodation, hotel, attractions, meals, location, longitude, latitude,
  visit_duration, category, ticket_price, type, name, estimated_cost,
  day_weather, night_weather, day_temp, night_temp, wind_direction, wind_power.
- Ensure the itinerary has exactly {travel_days} day(s), from {start_date} to {end_date}.

Original parse/validation error:
{error}

Original response:
{response}
""".strip()


@dataclass
class PlanningContext:
    attractions: list[POIInfo]
    hotels: list[Hotel]
    weather: list[WeatherInfo]
    web_context: str = ""


class TripPlannerAgent:
    """Trip planner that gathers map context and asks the shared LLM to plan."""

    def __init__(
        self,
        llm_client: Optional[HelloAgentsLLM] = None,
        map_service: Optional[AmapService] = None,
        use_web_fallback: Optional[bool] = None,
    ):
        self._llm_client = llm_client
        self.map_service = map_service or AmapService()
        if use_web_fallback is None:
            use_web_fallback = os.getenv("TRIP_PLANNER_USE_WEB_SEARCH", "").lower() in {
                "1",
                "true",
                "yes",
            }
        self.use_web_fallback = use_web_fallback

    @property
    def llm_client(self) -> HelloAgentsLLM:
        if self._llm_client is None:
            self._llm_client = get_llm()
        return self._llm_client

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """Generate a complete trip plan for a validated request."""
        print("\n=== Trip planning started ===")
        print(f"Destination: {request.city}")
        print(f"Dates: {request.start_date} to {request.end_date} ({request.travel_days} days)")
        if request.preferences:
            print(f"Preferences: {', '.join(request.preferences)}")

        context = self._collect_context(request)
        prompt = self._build_planner_prompt(request, context)

        try:
            response = self.llm_client.think_simple(prompt, temperature=0) or ""
        except Exception as exc:
            print(f"LLM planning failed: {exc}")
            response = ""

        if response:
            parsed = self._parse_response(response, request)
            if parsed is not None:
                print("=== Trip planning completed with LLM output ===\n")
                return parsed

            repaired = self._repair_response(response, request)
            if repaired is not None:
                print("=== Trip planning completed after JSON repair ===\n")
                return repaired

        print("Using deterministic fallback trip plan.")
        return self._create_fallback_plan(request, context)

    def _collect_context(self, request: TripRequest) -> PlanningContext:
        print("Collecting attraction context...")
        attractions = self._search_attractions(request)
        print(f"Collected {len(attractions)} attraction candidates.")

        print("Collecting weather context...")
        weather = self.map_service.get_weather(request.city, limit=request.travel_days)
        print(f"Collected {len(weather)} weather entries.")

        print("Collecting hotel context...")
        hotels = self.map_service.search_hotels(request.city, request.accommodation, limit=8)
        print(f"Collected {len(hotels)} hotel candidates.")

        web_context = ""
        if self.use_web_fallback and not attractions:
            web_context = self._web_search(
                f"best travel attractions in {request.city} " + " ".join(request.preferences)
            )

        return PlanningContext(
            attractions=attractions,
            hotels=hotels,
            weather=weather,
            web_context=web_context,
        )

    def _search_attractions(self, request: TripRequest) -> list[POIInfo]:
        keywords = self._attraction_keywords(request)
        seen: set[str] = set()
        attractions: list[POIInfo] = []

        for keyword in keywords:
            for poi in self.map_service.search_poi(keyword, request.city, limit=8):
                key = poi.id or f"{poi.name}|{poi.address}"
                if key in seen:
                    continue
                seen.add(key)
                attractions.append(poi)
                if len(attractions) >= max(8, request.travel_days * 3):
                    return attractions
        return attractions

    @staticmethod
    def _attraction_keywords(request: TripRequest) -> list[str]:
        if request.preferences:
            return [f"{preference} attraction" for preference in request.preferences] + ["tourist attraction"]
        return ["tourist attraction", "landmark", "museum"]

    def _build_planner_prompt(self, request: TripRequest, context: PlanningContext) -> str:
        return PLANNER_PROMPT_TEMPLATE.format(
            request_json=json.dumps(request.model_dump(), ensure_ascii=False, indent=2),
            attraction_context=self._format_pois(context.attractions),
            weather_context=self._format_weather(context.weather),
            hotel_context=self._format_hotels(context.hotels),
            web_context=context.web_context or "None",
        )

    @staticmethod
    def _format_pois(pois: list[POIInfo]) -> str:
        if not pois:
            return "No structured attraction data was available."
        lines = []
        for index, poi in enumerate(pois, start=1):
            lines.append(
                f"{index}. {poi.name} | category: {poi.type or 'unknown'} | "
                f"address: {poi.address or 'unknown'} | "
                f"coordinates: {poi.location.longitude}, {poi.location.latitude} | "
                f"id: {poi.id or 'unknown'}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_hotels(hotels: list[Hotel]) -> str:
        if not hotels:
            return "No structured hotel data was available."
        lines = []
        for index, hotel in enumerate(hotels, start=1):
            location = "unknown"
            if hotel.location:
                location = f"{hotel.location.longitude}, {hotel.location.latitude}"
            lines.append(
                f"{index}. {hotel.name} | type: {hotel.type or 'hotel'} | "
                f"address: {hotel.address or 'unknown'} | coordinates: {location}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_weather(weather: list[WeatherInfo]) -> str:
        if not weather:
            return "No structured weather data was available."
        return "\n".join(item.model_dump_json() for item in weather)

    def _parse_response(self, response: str, request: TripRequest) -> Optional[TripPlan]:
        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
            plan = TripPlan.model_validate(data)
            return self._normalize_plan(plan, request)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            print(f"Could not parse LLM trip plan JSON: {exc}")
            return None

    def _repair_response(self, response: str, request: TripRequest) -> Optional[TripPlan]:
        """Ask the shared LLM to repair malformed JSON once before falling back."""
        prompt = JSON_REPAIR_PROMPT_TEMPLATE.format(
            travel_days=request.travel_days,
            start_date=request.start_date,
            end_date=request.end_date,
            error="The previous response did not parse or validate as TripPlan JSON.",
            response=response,
        )
        try:
            repaired_response = self.llm_client.think_simple(prompt, temperature=0) or ""
        except Exception as exc:
            print(f"LLM JSON repair failed: {exc}")
            return None
        if not repaired_response:
            return None
        return self._parse_response(repaired_response, request)

    @staticmethod
    def _extract_json(text: str) -> str:
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found")
        return text[start : end + 1]

    def _normalize_plan(self, plan: TripPlan, request: TripRequest) -> TripPlan:
        """Make small deterministic corrections after LLM generation."""
        dates = self._date_strings(request)
        plan.city = request.city
        plan.start_date = request.start_date
        plan.end_date = request.end_date

        for index, day in enumerate(plan.days[: request.travel_days]):
            day.day_index = index
            day.date = dates[index]
            if not day.transportation:
                day.transportation = request.transportation
            if not day.accommodation:
                day.accommodation = request.accommodation
        plan.days = plan.days[: request.travel_days]

        if len(plan.days) != request.travel_days:
            raise ValueError("LLM returned the wrong number of days")

        if plan.budget is None:
            plan.budget = self._calculate_budget(plan)
        return plan

    def _create_fallback_plan(self, request: TripRequest, context: Optional[PlanningContext] = None) -> TripPlan:
        dates = self._date_strings(request)
        attractions = context.attractions if context else []
        hotels = context.hotels if context else []
        weather = context.weather if context else []

        if not attractions:
            city_center = self.map_service.geocode(request.city) or Location(longitude=0.0, latitude=0.0)
            attractions = [
                POIInfo(
                    id=f"fallback-{index}",
                    name=name,
                    type="suggested area",
                    address=request.city,
                    location=city_center,
                )
                for index, name in enumerate(
                    [
                        f"{request.city} historic center",
                        f"{request.city} landmark district",
                        f"{request.city} local food neighborhood",
                        f"{request.city} museum or cultural stop",
                    ],
                    start=1,
                )
            ]

        default_hotel = Hotel(
            name=f"Recommended {request.accommodation} area in {request.city}",
            address=request.city,
            location=attractions[0].location if attractions else None,
            type=request.accommodation,
            estimated_cost=120,
        )

        days: list[DayPlan] = []
        for day_index, date_text in enumerate(dates):
            day_pois = [
                attractions[(day_index * 2 + offset) % len(attractions)]
                for offset in range(min(2, len(attractions)))
            ]
            hotel = hotels[day_index % len(hotels)] if hotels else default_hotel
            if hotel.estimated_cost == 0:
                hotel.estimated_cost = 120

            days.append(
                DayPlan(
                    date=date_text,
                    day_index=day_index,
                    description=f"Day {day_index + 1}: explore a balanced mix of highlights in {request.city}.",
                    transportation=request.transportation,
                    accommodation=request.accommodation,
                    hotel=hotel,
                    attractions=[self._poi_to_attraction(poi) for poi in day_pois],
                    meals=self._fallback_meals(day_index),
                )
            )

        plan = TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=weather or self._fallback_weather(dates),
            overall_suggestions=(
                "This fallback itinerary was generated with limited live context. "
                "Confirm opening hours, ticket prices, weather, and transport times before booking."
            ),
        )
        plan.budget = self._calculate_budget(plan)
        return plan

    @staticmethod
    def _poi_to_attraction(poi: POIInfo) -> Attraction:
        return Attraction(
            name=poi.name,
            address=poi.address,
            location=poi.location,
            visit_duration=120,
            description=f"Visit {poi.name} and leave time to explore the surrounding area.",
            category=poi.type or "attraction",
            poi_id=poi.id,
            ticket_price=0,
        )

    @staticmethod
    def _fallback_meals(day_index: int) -> list[Meal]:
        day = day_index + 1
        return [
            Meal(
                type="breakfast",
                name=f"Day {day} local breakfast near the hotel",
                description="Choose a nearby cafe or local breakfast spot before sightseeing.",
                estimated_cost=15,
            ),
            Meal(
                type="lunch",
                name=f"Day {day} casual lunch near the main attractions",
                description="Keep lunch close to the day's route to reduce transit time.",
                estimated_cost=25,
            ),
            Meal(
                type="dinner",
                name=f"Day {day} regional dinner",
                description="Try a well-reviewed restaurant featuring local specialties.",
                estimated_cost=35,
            ),
        ]

    @staticmethod
    def _fallback_weather(dates: list[str]) -> list[WeatherInfo]:
        return [
            WeatherInfo(
                date=date_text,
                day_weather="Check forecast",
                night_weather="Check forecast",
                day_temp=0,
                night_temp=0,
                wind_direction="",
                wind_power="",
            )
            for date_text in dates
        ]

    @staticmethod
    def _calculate_budget(plan: TripPlan) -> Budget:
        total_attractions = sum(
            attraction.ticket_price for day in plan.days for attraction in day.attractions
        )
        total_hotels = sum(day.hotel.estimated_cost for day in plan.days if day.hotel)
        total_meals = sum(meal.estimated_cost for day in plan.days for meal in day.meals)
        total_transportation = 20 * len(plan.days)
        return Budget(
            total_attractions=total_attractions,
            total_hotels=total_hotels,
            total_meals=total_meals,
            total_transportation=total_transportation,
            total=total_attractions + total_hotels + total_meals + total_transportation,
        )

    @staticmethod
    def _date_strings(request: TripRequest) -> list[str]:
        start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        return [
            (start + timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in range(request.travel_days or 0)
        ]

    @staticmethod
    def _web_search(query: str) -> str:
        """Optional web-search fallback using the existing src/WebSearch function."""
        project_root = Path(__file__).resolve().parents[1]
        src_path = project_root / "src"
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        try:
            from WebSearch import ddgs_search  # noqa: WPS433

            result = ddgs_search(query)
            if result.startswith("Error"):
                return ""
            return result
        except Exception as exc:
            print(f"Optional web-search fallback failed: {exc}")
            return ""


_multi_agent_planner: Optional[TripPlannerAgent] = None


# Backward-compatible name matching the upstream example.
MultiAgentTripPlanner = TripPlannerAgent


def get_trip_planner_agent() -> TripPlannerAgent:
    """Return a singleton trip planner agent."""
    global _multi_agent_planner
    if _multi_agent_planner is None:
        _multi_agent_planner = TripPlannerAgent()
    return _multi_agent_planner
