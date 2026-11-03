"""Convert predicted traffic flow into travel time.

OWNER: Hassan (Calculation of time)
Source: docs/Traffic_Flow_to_Travel_Time_Conversion_v1.0.pdf

Model (simplified fundamental diagram):

    flow = a * speed^2 + b * speed,   a = -1.4648375, b = 93.75

  - Turning point (capacity): flow = 1500 veh/h at speed = 32 km/h.
  - Under capacity (green branch) -> take the HIGHER root (speed >= 32).
  - Over capacity (red branch)    -> the LOWER root. The PDF lets us assume
    every segment is under capacity, so use the green branch; flows above 1500
    should be clamped (document it).
  - If the computed speed is above the speed limit (flow <= ~351 veh/h),
    cap the speed at 60 km/h.

Edge travel time (seconds) = distance_km / speed_kmh * 3600 + intersection_delay (30 s)
"""
from __future__ import annotations

from src.config import load_config


def flow_to_speed(flow_vph: float, cfg: dict | None = None) -> float:
    """Return the expected speed in km/h for a flow in vehicles/hour.

    Expected values (see tests/test_travel_time.py):
        flow_to_speed(0)    == 60.0
        flow_to_speed(351)  == 60.0      (still capped)
        flow_to_speed(1500) == ~32.0     (capacity)
    TODO(Hassan): solve the quadratic, pick the green-branch root, cap at the limit.
    """
    cfg = cfg or load_config()
    raise NotImplementedError("Hassan: implement flow_to_speed()")


def edge_travel_time(distance_km: float, flow_vph: float, cfg: dict | None = None) -> float:
    """Travel time in SECONDS for one edge, including the intersection delay.

    Example: 1 km at flow 100 veh/h -> 60 km/h -> 60 s + 30 s = 90 s
    TODO(Hassan)
    """
    cfg = cfg or load_config()
    raise NotImplementedError("Hassan: implement edge_travel_time()")
