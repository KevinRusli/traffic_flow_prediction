# Data notes - SCATS October 2006 (Boroondara)

Things we found by inspecting `data/raw/Scats Data October 2006.xls`. Read this before writing the loader.

## Shape

| Item | Value |
|---|---|
| Sheets | `Notes`, `Data`, `Summary Of Data` -> use **`Data`** |
| Header | The real column names are on the **2nd row** -> `pd.read_excel(..., sheet_name="Data", header=1)` |
| Rows | 4,192 (one row = one approach location x one day) |
| SCATS sites (intersections = graph nodes) | **40** |
| Locations (detector approaches, e.g. `WARRIGAL_RD N of TOORAK_RD`) | **139** -> ~3-4 per site |
| Date range | 2006-10-01 to 2006-10-31 |
| Volume columns | `V00` ... `V95` = 96 counts of 15 min (V00 = 00:00-00:15) |

## Gotchas

1. **Date column has a 00:15 time** (`2006-10-01 00:15:00`). Normalise it to midnight before adding the interval offset, otherwise every timestamp is 15 min late.
2. **Two detectors, one name**: location `HIGH_ST NE of CHARLES_ST` (site **4335**) has 62 rows instead of 31. They are NOT duplicates: `VR Internal Loc` is 2 for one set and 6 for the other, and the counts differ (e.g. 7,423 vs 8,850 vehicles on 1 Oct). The loader keeps both and renames the second one `HIGH_ST NE of CHARLES_ST #2` -> 140 locations.
3. **Missing days**: 34 locations have fewer than 31 days (e.g. `CHURCH_ST SW of BARKERS_RD` has only **2** days, many have 22-30). `build_dataset` splits each location into continuous stretches and never builds a window across a gap.
4. **Zero coordinates**: `AUBURN_RD N of BURWOOD_RD` (site **4266**) has `NB_LATITUDE = NB_LONGITUDE = 0` in all 31 rows. The other three approaches of 4266 have coordinates, so the loader uses their median. (If a whole site were missing it would fall back to the Traffic_Count CSV, matching the two road names.)
5. **Coordinate offset**: the spec says the lat/lon don't match the real intersections. We compared the dataset with the real-world coordinates in `Traffic_Count_Locations_with_LONG_LAT.csv`:

   | Site | Real - dataset (lat, lon) |
   |---|---|
   | 2000 Warrigal/Toorak | +0.00152, +0.00125 |
   | 3002 Denmark/Barkers | +0.00145, +0.00136 |
   | 970 Warrigal/High St | +0.00183, +0.00112 |
   | 4266 Auburn/Burwood | +0.00146, +0.00193 |

   So the loader adds **+0.0015 lat, +0.0013 lon** (`config.yaml` -> `data.coord_offset`). Good evidence for the report.
6. **Road-name inconsistencies**: `HIGH STREET_RD` vs `HIGH_ST`, `OF` vs `of`, directions `N/S/E/W/NE/NW/SE/SW`. Normalise before matching sites that share a road.
7. **Useless columns**: `CD_MELWAY`, `HF VicRoads Internal`, `VR Internal Stat`, `VR Internal Loc`, `NB_TYPE_SURVEY` - drop them.
8. Raw data warning from VicRoads: about 10% of data may be unusable (sensor faults, over-counting). Look for suspicious zeros / spikes in EDA.

## The 40 SCATS sites

```
 970 2000 2200 2820 2825 2827 2846 3001 3002 3120
3122 3126 3127 3180 3662 3682 3685 3804 3812 4030
4032 4034 4035 4040 4043 4051 4057 4063 4262 4263
4264 4266 4270 4272 4273 4321 4324 4335 4812 4821
```

Spec example: origin **2000** (WARRIGAL_RD/TOORAK_RD) -> destination **3002** (DENMARK_ST/BARKERS_RD).

## Map

`docs/Boroondara_SCATS_map.pdf` shows where every site is - use it to check the inferred edges.

## Result of the pipeline (`load_long()` + `build_dataset()`, lags = 12)

| Item | Value |
|---|---|
| Long table | 402,432 rows (4,192 location-days x 96) |
| Train samples (targets 1-24 Oct) | 314,376 |
| Test samples (targets 25-31 Oct) | 86,016 |
| Scaler | MinMax, fitted on train flows only (test max = 0.886, i.e. no test value above the train max) |
| Time to build from cache | < 1 s (Excel parsed once, cached in `data/processed/scats_long.csv`) |

## Road graph (`src/graph/build_graph.py`)

Links are inferred from the location names: two sites are linked when they lie on the same road
and no other site lies between them along that road. Result: 40 nodes, 52 two-way links
(`results/figures/road_graph.png`). Corrections, checked against `Boroondara_SCATS_map.pdf`:

| Case | Decision |
|---|---|
| `HARP_ST` (4321) | same road as `HARP_RD` (4321 uses both names) |
| `MAROONDAH_HWY` (2200) | continuation of Whitehorse Rd on the map |
| `VICTORIA_ST` (4821) | continuation of Barkers Rd on the map |
| `BURWOOD_HWY` (2000) | continuation of Toorak Rd |
| Eastern Fwy ramp names (2825, 2827) | one road, Eastern Fwy |
| `HIGH_ST` at 2846 | High St, Glen Iris - a different road from High St, Kew |
| 4812, 2846 share no road with another site | linked to their nearest site |

Edge length = straight-line distance, so real driving distances on curved roads are slightly longer.
