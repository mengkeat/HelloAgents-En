"""Command-line entry point for the trip planner agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from .agent import get_trip_planner_agent
from .models import TripRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an English travel itinerary with the trip planner agent.")
    parser.add_argument("--city", required=True, help="Destination city")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--days", type=int, default=None, help="Trip length; derived from dates when omitted")
    parser.add_argument("--transportation", default="public transit", help="Preferred transportation mode")
    parser.add_argument("--accommodation", default="mid-range hotel", help="Accommodation preference")
    parser.add_argument(
        "--preference",
        dest="preferences",
        action="append",
        default=[],
        help="Travel preference tag. Repeat for multiple preferences.",
    )
    parser.add_argument("--note", default="", help="Additional free-form requirement")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        request = TripRequest(
            city=args.city,
            start_date=args.start_date,
            end_date=args.end_date,
            travel_days=args.days,
            transportation=args.transportation,
            accommodation=args.accommodation,
            preferences=args.preferences,
            free_text_input=args.note,
        )
    except ValidationError as exc:
        print(f"Invalid trip request:\n{exc}")
        return 2

    plan = get_trip_planner_agent().plan_trip(request)
    output = plan.model_dump_json(indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Trip plan written to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
