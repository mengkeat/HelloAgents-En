"""Small Amap REST client used by the trip planner agent.

The upstream example uses an Amap MCP server. This local port avoids adding a new
agent framework dependency and calls the public REST endpoints directly with
`requests`, which is already managed by this uv project.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from .models import Hotel, Location, POIInfo, WeatherInfo

load_dotenv()


class AmapService:
    """Wrapper around a subset of the Amap Web Service API."""

    base_url = "https://restapi.amap.com/v3"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key or os.getenv("AMAP_API_KEY") or os.getenv("AMAP_MAPS_API_KEY") or ""
        self.timeout = timeout
        self._warned_missing_key = False

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search_poi(self, keywords: str, city: str, limit: int = 10, citylimit: bool = True) -> list[POIInfo]:
        """Search for points of interest by keyword and city."""
        if not self._has_key("POI search"):
            return []

        data = self._get(
            "place/text",
            {
                "keywords": keywords,
                "city": city,
                "citylimit": "true" if citylimit else "false",
                "offset": min(max(limit, 1), 25),
                "page": 1,
                "extensions": "base",
            },
        )
        pois = data.get("pois", []) if data else []
        results: list[POIInfo] = []
        for raw in pois[:limit]:
            location = self._parse_location(raw.get("location"))
            if location is None:
                continue
            address = raw.get("address") or ""
            if isinstance(address, list):
                address = ", ".join(str(item) for item in address)
            results.append(
                POIInfo(
                    id=str(raw.get("id") or ""),
                    name=str(raw.get("name") or "Unknown place"),
                    type=str(raw.get("type") or ""),
                    address=str(address),
                    location=location,
                    tel=str(raw.get("tel") or "") or None,
                )
            )
        return results

    def search_hotels(self, city: str, accommodation: str = "hotel", limit: int = 8) -> list[Hotel]:
        """Search for hotels and map the results to Hotel models."""
        keywords = accommodation if accommodation else "hotel"
        if "hotel" not in keywords.lower():
            keywords = f"{keywords} hotel"

        return [self._poi_to_hotel(poi, accommodation) for poi in self.search_poi(keywords, city, limit=limit)]

    def get_weather(self, city: str, limit: Optional[int] = None) -> list[WeatherInfo]:
        """Return forecast weather for a city when available."""
        if not self._has_key("weather lookup"):
            return []

        data = self._get("weather/weatherInfo", {"city": city, "extensions": "all"})
        casts = []
        forecasts = data.get("forecasts", []) if data else []
        if forecasts:
            casts = forecasts[0].get("casts", []) or []

        weather: list[WeatherInfo] = []
        for cast in casts[:limit]:
            weather.append(
                WeatherInfo(
                    date=str(cast.get("date") or ""),
                    day_weather=str(cast.get("dayweather") or ""),
                    night_weather=str(cast.get("nightweather") or ""),
                    day_temp=cast.get("daytemp") or 0,
                    night_temp=cast.get("nighttemp") or 0,
                    wind_direction=str(cast.get("daywind") or ""),
                    wind_power=str(cast.get("daypower") or ""),
                )
            )
        return weather

    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """Resolve an address to coordinates."""
        if not self._has_key("geocoding"):
            return None

        params: dict[str, Any] = {"address": address}
        if city:
            params["city"] = city
        data = self._get("geocode/geo", params)
        geocodes = data.get("geocodes", []) if data else []
        if not geocodes:
            return None
        return self._parse_location(geocodes[0].get("location"))

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        request_params = {"key": self.api_key, **params}
        try:
            response = requests.get(url, params=request_params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"Map API request failed for {endpoint}: {exc}")
            return {}

        if str(data.get("status")) != "1":
            info = data.get("info") or data.get("infocode") or "unknown error"
            print(f"Map API returned an error for {endpoint}: {info}")
            return {}
        return data

    def _has_key(self, feature: str) -> bool:
        if self.api_key:
            return True
        if not self._warned_missing_key:
            print(
                "AMAP_API_KEY is not configured; "
                f"{feature} will be skipped and the planner will use fallback context."
            )
            self._warned_missing_key = True
        return False

    @staticmethod
    def _parse_location(value: Any) -> Optional[Location]:
        if not value or not isinstance(value, str) or "," not in value:
            return None
        lon_text, lat_text = value.split(",", 1)
        try:
            return Location(longitude=float(lon_text), latitude=float(lat_text))
        except ValueError:
            return None

    @staticmethod
    def _poi_to_hotel(poi: POIInfo, accommodation: str) -> Hotel:
        return Hotel(
            name=poi.name,
            address=poi.address,
            location=poi.location,
            type=accommodation,
        )
