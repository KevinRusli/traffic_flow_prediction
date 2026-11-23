"""Convert predicted traffic flow into travel time.

OWNER: Hassan (Calculation of time)
Source: docs/Traffic_Flow_to_Travel_Time_Conversion_v1.0.pdf

Model (simplified fundamental diagram):

    flow = a * speed^2 + b * speed,   a = -1.4648375, b = 93.75

  - Turning point (capacity): flow = 1500 veh/h at speed = 32 km/h.
  - Under capacity (green branch) -> take the HIGHER root (speed >= 32).
  - Over capacity (red branch)    -> the LOWER root. The PDF lets us assume
    every segment is under capacity, so we always use the green branch and
    clamp flows above capacity to 1500 veh/h (speed = 32 km/h).
  - If the computed speed is above the speed limit (flow <= ~351 veh/h),
    cap the speed at 60 km/h.

Edge travel time (seconds) = distance_km / speed_kmh * 3600 + intersection_delay (30 s)
"""
from __future__ import annotations

import math

from src.config import load_config


def flow_to_speed(flow_vph: float, cfg: dict | None = None) -> float:
    """Return the expected speed in km/h for a flow in vehicles/hour.

    Expected values (see tests/test_travel_time.py):
        flow_to_speed(0)    == 60.0
        flow_to_speed(351)  == 60.0      (still capped)
        flow_to_speed(1500) == ~32.0     (capacity)
    """
    cfg = cfg or load_config()
    rcfg = cfg["routing"]
    a, b = rcfg["flow_speed_a"], rcfg["flow_speed_b"]

    # A model can predict a tiny negative flow; above capacity we assume the
    # road is still at capacity (PDF: "the traffic at each segment is under capacity").
    flow = min(max(flow_vph, 0.0), rcfg["capacity_flow_vph"])

    # Solve a*s^2 + b*s - flow = 0 with the quadratic formula.
    # At capacity the discriminant is ~0 but rounding can make it slightly negative.
    disc = max(b * b + 4 * a * flow, 0.0)
    # a < 0, so "-sqrt" gives the higher root = green (under capacity) branch.
    speed = (-b - math.sqrt(disc)) / (2 * a)

    # Low flow gives a speed above the limit -> drivers still go at the limit.
    return min(speed, rcfg["speed_limit_kmh"])


def edge_travel_time(distance_km: float, flow_vph: float, cfg: dict | None = None) -> float:
    """Travel time in SECONDS for one edge, including the intersection delay.

    Example: 1 km at flow 100 veh/h -> 60 km/h -> 60 s + 30 s = 90 s
    """
    cfg = cfg or load_config()
    speed = flow_to_speed(flow_vph, cfg)
    driving_s = distance_km / speed * 3600          # hours -> seconds
    return driving_s + cfg["routing"]["intersection_delay_s"]
