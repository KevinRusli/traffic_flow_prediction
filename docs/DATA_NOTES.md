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
2. **Duplicate rows**: location `HIGH_ST NE of CHARLES_ST` (site **4335**) has 62 rows instead of 31. Drop duplicates.
3. **Missing days**: 34 locations have fewer than 31 days (e.g. `CHURCH_ST SW of BARKERS_RD` has only **2** days, many have 22-30). Never build a sliding window across a gap.
4. **Zero coordinates**: site **4266** has `NB_LATITUDE = NB_LONGITUDE = 0` in all 31 rows. Fill from another source (site listing / Traffic_Count CSV / the map PDF).
5. **Coordinate offset**: the spec says the lat/lon don't match the real intersections on Google Maps. A commonly used correction is about **+0.0015 lat, +0.0013 lon** (in `config.yaml` -> `data.coord_offset`). Verify visually and describe what you did in the report.
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
