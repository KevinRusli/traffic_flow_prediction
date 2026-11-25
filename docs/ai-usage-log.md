# AI usage log

The assignment requires the report to state how AI was used (or not used).
Each member adds a row after every session that involved an AI tool.
Be specific: what you asked, what you used, what you changed or rejected.

| Date | Member | Tool | What it was used for | What we kept / changed |
|---|---|---|---|---|
| 2026-11-03 | Kevin | Claude | Read the A2B spec and datasets; drafted the repository template (folder structure, module contracts, config, test scaffolding, CI, docs); wrote the A2A adapter (`TravelTimeProblem`), Yen top-k and `RouteFinder`, plus their tests | Reviewed the template; A2A algorithms kept unchanged; task split taken from the tutor's list |
| 2026-11-03 | Kevin | Claude | Data processing: inspected the dataset quirks (4335 = two detectors, 4266 missing coordinates, coordinate offset checked against the Traffic_Count CSV); wrote `loader.py`, `preprocess.py`, `parse_location` and their tests | Decided to keep both 4335 detectors and to split train/test by target date; reviewed every function |
| 2026-11-03 | Kevin | Claude | Road graph inference (`build_graph`, checked against the VicRoads map), `ModelPredictor`, tests TC08-TC09 and TC30, tidied comments in Kevin's modules | Road-name corrections decided by comparing with the map; checked an end-to-end route 2000 -> 3002 |
| 2026-11-09 | Kevin | Claude | Reviewed Kevin's modules against the spec (ran pytest, flake8 and a real-data check of 2000 -> 3002 with every 2A method); simplified the road ordering in `build_graph` (sort by the wider axis instead of SVD, same 52 links), added missing comments/docstrings, friendly CLI errors in `main.py route`, fixed the README markdown escapes | Checked the graph is unchanged and all tests still pass; aggregation mean vs sum left for the tutor to confirm |
| 2026-11-19 | Ali Murtaza | Claude | Got help understanding GRU gate mechanics vs LSTM, and how Streamlit/pydeck fit together for the GUI; asked for a nudge when stuck on input reshaping and route/map wiring | Wrote and tested the final code myself (`src/models/gru.py`, `app/gui.py`); checked it with `pytest -k gru` and by running the GUI locally |
