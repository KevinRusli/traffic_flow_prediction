# GUI spec

OWNER: Ali Murtaza - file: `app/gui.py`

**Tutor's direction:** like a simple Google Maps. For every route show **how many minutes in total**
and **which roads it takes**. Do not over-engineer it.

## Layout

```
+--------------------+-------------------------------------------------+
| Origin   [2000   ] |                                                 |
| Dest.    [3002   ] |        MAP (OpenStreetMap base)                 |
| Date     [15 Oct ] |   - every SCATS site as a dot                   |
| Time     [08:00  ] |   - origin / destination highlighted            |
| Model    [GRU   v] |   - route 1 = thick coloured line               |
| Search   [AS    v] |   - routes 2..5 = thin grey lines               |
| Routes   [===5   ] |                                                 |
| [ Find routes ]    |                                                 |
+--------------------+-------------------------------------------------+
| Route 1  [Fastest]                                          14 min    |
| Warrigal Rd -> Riversdale Rd -> Burke Rd -> Barkers Rd                 |
| 6.2 km - 5 intersections                                              |
+-----------------------------------------------------------------------+
| Route 2                                                     17 min    |
| Toorak Rd -> Burke Rd -> Barkers Rd                                   |
| 6.8 km - 6 intersections                                              |
+-----------------------------------------------------------------------+
```

## Must have (spec + tutor)

- Inputs: origin, destination, date + time, model, search method, number of routes (1-5)
- Defaults come from `config.yaml` (`gui:` and `routing:` sections)
- Up to 5 routes, fastest first, each with **total minutes**, **road names**, distance
- Map with the routes drawn
- Friendly errors: unknown SCATS site, no route found

## Nice to have (only if time allows)

- Expander per route with the segment table (road, from site, to site, minutes)
- Second tab with Sakib's model comparison (`results/metrics.csv` + figures)
- Colour roads by predicted congestion (this is a Research idea, coordinate with Sakib)

## Data you get

```python
routes = finder.find_routes(origin, destination, when, k=k, method=method)
route.travel_time_min      # 14.2
route.distance_km          # 6.2
route.roads                # ["Warrigal Rd", "Riversdale Rd", "Burke Rd", "Barkers Rd"]
route.path                 # [2000, 4043, 3180, 3002]  SCATS sites
route.segments             # Segment(src, dst, road, distance_km, time_s) per edge
finder.graph.nodes[2000].lat, .lon
```

Develop with `ConstantPredictor` (already working) so you don't wait for the ML models.
