# Rules for AI assistants

Read this before changing anything in this repository. It is the A2B version of our 2A rules.

## Scope

- Only edit the files owned by the person you are helping (see the ownership table in `README.md`).
- **Never edit `src/search/a2a/`.** It is our Assignment 2A code, reused unchanged through
  `src/search/adapter.py`. Changing it breaks the "Part A integrated directly" claim in the report.
- Do not change a shared contract (`src/graph/graph.py`, `src/models/base.py`,
  `src/prediction/predictor.py`, column names in `docs/INTERFACES.md`, `config.yaml` keys)
  unless the user says the whole team agreed.
- Never copy code from `reference/` (the tutor's examples) into `src/`.

## Contracts to respect

- Data: `load_long()` returns columns `scats, location, lat, lon, timestamp, flow`.
- Models: subclass `BaseTrafficModel` (`fit`, `predict`, `save`, `load`); 2-D input `(n, lags)`.
- Predictor: `predict_flow(scats, when) -> vehicles per hour`.
- Search: 2A contract `search(problem) -> (goal, nodes_created, path)`.
- Routing: `RouteFinder.find_routes(...) -> list[Route]` with `travel_time_min`, `distance_km`, `roads`.

## Tests

- Run `pytest` after every change. Tests raising `NotImplementedError` show as XFAIL; once a
  function is implemented its tests must PASS.
- Never weaken or delete a test to make it pass. Never edit expected numbers that come from the spec.

## Git

- Work on `main`, `git pull` before `git push`.
- **Do not add `Co-Authored-By:` or any AI attribution / session line to commit messages or PRs.**

## Code quality (marked, up to -10 for poor quality)

- A module docstring saying what the file does and who owns it.
- Short comments on every non-obvious step; clear names; small functions; no dead code.
- The user must be able to explain every line in the report and in class, so prefer simple code
  over clever code, and explain your choices when asked.

## AI use must be acknowledged

The report must state how AI was used. Remind the user to add a row to `docs/ai-usage-log.md`
after every session that used AI.
