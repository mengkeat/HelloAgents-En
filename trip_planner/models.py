"""Data models for the English trip planner agent port."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _parse_iso_date(value: str):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("dates must use YYYY-MM-DD format") from exc


class TripRequest(BaseModel):
    """User request for a travel plan."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Beijing",
                "start_date": "2026-06-01",
                "end_date": "2026-06-03",
                "travel_days": 3,
                "transportation": "public transit",
                "accommodation": "mid-range hotel",
                "preferences": ["history", "local food"],
                "free_text_input": "Prefer museums and avoid very late nights.",
            }
        }
    )

    city: str = Field(..., min_length=1, description="Destination city")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    travel_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Trip length in calendar days. Derived from dates when omitted.",
    )
    transportation: str = Field(default="public transit", description="Preferred transportation mode")
    accommodation: str = Field(default="mid-range hotel", description="Accommodation preference")
    preferences: list[str] = Field(default_factory=list, description="Travel style tags")
    free_text_input: str = Field(default="", description="Additional free-form requirements")

    @field_validator("city", "start_date", "end_date", "transportation", "accommodation", "free_text_input")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("preferences")
    @classmethod
    def clean_preferences(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]

    @model_validator(mode="after")
    def validate_dates(self) -> "TripRequest":
        start = _parse_iso_date(self.start_date)
        end = _parse_iso_date(self.end_date)
        if end < start:
            raise ValueError("end_date must be on or after start_date")

        computed_days = (end - start).days + 1
        if self.travel_days is None:
            self.travel_days = computed_days
        elif self.travel_days != computed_days:
            raise ValueError(
                "travel_days must match the inclusive date range "
                f"({computed_days} days for {self.start_date} to {self.end_date})"
            )
        return self


class Location(BaseModel):
    """Geographic coordinates."""

    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")


class Attraction(BaseModel):
    """Attraction included in a daily itinerary."""

    name: str = Field(..., description="Attraction name")
    address: str = Field(default="", description="Street address or area")
    location: Location = Field(..., description="Coordinates")
    visit_duration: int = Field(default=120, ge=0, description="Suggested visit duration in minutes")
    description: str = Field(default="", description="Why this stop is included")
    category: str = Field(default="attraction", description="Attraction category")
    rating: Optional[float] = Field(default=None, description="Rating when available")
    photos: list[str] = Field(default_factory=list, description="Photo URLs")
    poi_id: str = Field(default="", description="Provider POI identifier")
    image_url: Optional[str] = Field(default=None, description="Primary image URL")
    ticket_price: int = Field(default=0, ge=0, description="Estimated ticket price in local currency")


class Meal(BaseModel):
    """Meal recommendation."""

    type: str = Field(..., description="breakfast, lunch, dinner, or snack")
    name: str = Field(..., description="Meal recommendation name")
    address: Optional[str] = Field(default=None, description="Address when known")
    location: Optional[Location] = Field(default=None, description="Coordinates when known")
    description: Optional[str] = Field(default=None, description="Recommendation details")
    estimated_cost: int = Field(default=0, ge=0, description="Estimated cost in local currency")


class Hotel(BaseModel):
    """Hotel recommendation."""

    name: str = Field(..., description="Hotel name")
    address: str = Field(default="", description="Hotel address")
    location: Optional[Location] = Field(default=None, description="Hotel coordinates")
    price_range: str = Field(default="", description="Price range when known")
    rating: str = Field(default="", description="Rating when known")
    distance: str = Field(default="", description="Distance to relevant sights when known")
    type: str = Field(default="", description="Accommodation type")
    estimated_cost: int = Field(default=0, ge=0, description="Estimated cost per night in local currency")


class DayPlan(BaseModel):
    """Plan for one day of the trip."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    day_index: int = Field(..., ge=0, description="Zero-based day index")
    description: str = Field(..., description="Summary for the day")
    transportation: str = Field(..., description="Transportation recommendation")
    accommodation: str = Field(..., description="Accommodation recommendation")
    hotel: Optional[Hotel] = Field(default=None, description="Suggested hotel")
    attractions: list[Attraction] = Field(default_factory=list, description="Stops for the day")
    meals: list[Meal] = Field(default_factory=list, description="Meal recommendations")


class WeatherInfo(BaseModel):
    """Daily weather summary."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    day_weather: str = Field(default="", description="Daytime weather")
    night_weather: str = Field(default="", description="Night weather")
    day_temp: Union[int, str] = Field(default=0, description="Daytime temperature")
    night_temp: Union[int, str] = Field(default=0, description="Night temperature")
    wind_direction: str = Field(default="", description="Wind direction")
    wind_power: str = Field(default="", description="Wind force")

    @field_validator("day_temp", "night_temp", mode="before")
    @classmethod
    def parse_temperature(cls, value):
        if isinstance(value, str):
            cleaned = value.replace("°C", "").replace("℃", "").replace("°", "").strip()
            try:
                return int(float(cleaned))
            except ValueError:
                return 0
        return value


class Budget(BaseModel):
    """Trip budget estimate."""

    total_attractions: int = Field(default=0, ge=0, description="Attraction ticket total")
    total_hotels: int = Field(default=0, ge=0, description="Hotel total")
    total_meals: int = Field(default=0, ge=0, description="Meal total")
    total_transportation: int = Field(default=0, ge=0, description="Transportation total")
    total: int = Field(default=0, ge=0, description="Grand total")


class TripPlan(BaseModel):
    """Complete travel plan."""

    city: str = Field(..., description="Destination city")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")
    days: list[DayPlan] = Field(..., description="Daily itinerary")
    weather_info: list[WeatherInfo] = Field(default_factory=list, description="Daily weather")
    overall_suggestions: str = Field(..., description="General advice")
    budget: Optional[Budget] = Field(default=None, description="Budget estimate")


class POIInfo(BaseModel):
    """Point-of-interest returned by a map provider."""

    id: str = Field(default="", description="Provider POI ID")
    name: str = Field(..., description="POI name")
    type: str = Field(default="", description="POI category")
    address: str = Field(default="", description="POI address")
    location: Location = Field(..., description="POI coordinates")
    tel: Optional[str] = Field(default=None, description="Phone number")
