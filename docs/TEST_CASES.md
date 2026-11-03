# Test cases

Rubric: **at least 15 test cases covering different scenarios, results checked and documented** (10 marks).
The pytest suite already contains 28 numbered cases. Each owner fills the *Result* column once their part passes,
and Kevin (test lead) copies the final table into the report's Testing section.

Run everything: `pytest`  - one file: `pytest tests/test_search.py -v`

Unimplemented code shows as **XFAIL** ("not implemented yet"); it flips to PASS/FAIL automatically once implemented.

| ID | Area | Scenario | Expected | Owner | Result |
|---|---|---|---|---|---|
| TC01 | Data | Load raw workbook | 40 sites, V00-V95 present, no duplicate rows | Kevin | |
| TC02 | Data | Wide -> long | 96 rows per day, 15-min spacing, flow >= 0 | Kevin | |
| TC03 | Data | Fix coordinates | No 0 lat/lon (site 4266 fixed) | Kevin | |
| TC04 | Data | Sliding windows toy series | `[1..5]`, lags 3 -> 2 windows | Kevin | |
| TC05 | Data | Chronological split | max(train time) < min(test time) | Kevin | |
| TC06 | Data | build_dataset | correct shapes, train scaled to [0,1] | Kevin | |
| TC07 | Graph | Parse location strings | `"HIGH_ST NE OF HARP_ST"` -> (road, dir, cross road) | Kevin | |
| TC08 | Graph | Real graph | 40 nodes, every node has an edge | Kevin | |
| TC09 | Graph | Connectivity | 2000 -> 3002 reachable | Kevin | |
| TC10 | Graph | Shared RoadGraph class | adjacency + haversine work | Kevin | PASS |
| TC11 | Models | fit/predict for lstm, gru, third | output shape (n,), finite | Hassan / Ali / Sakib | |
| TC12 | Models | save -> load round-trip | identical predictions | Hassan / Ali / Sakib | |
| TC13 | Eval | Metrics on known values | RMSE = sqrt(3), NRMSE = sqrt(3)/20, perfect R2 = 1 | Sakib | |
| TC14 | Travel time | Flow -> speed reference points | 0/100/351 -> 60 km/h, 1500 -> 32 km/h | Hassan | |
| TC15 | Travel time | 1 km at low flow | 60 s + 30 s delay = 90 s | Hassan | |
| TC16 | Travel time | Monotonic | speed decreases from 400 to 1500 veh/h | Hassan | |
| TC17 | Search | 2A AS through the adapter | optimal path 1-2-4, cost 2 | Kevin | PASS |
| TC18 | Search | All 6 unchanged 2A methods via adapter | valid path start->goal along real edges | Kevin | PASS |
| TC19 | Search | Yen top-k | sorted, distinct, loopless, <= k | Kevin | PASS |
| TC20 | Search | Unreachable goal | `None` / `[]`, no crash | Kevin | PASS |
| TC21 | Routing | RouteFinder + ConstantPredictor | best route 1-2-4 = 180 s, via A Rd | Kevin (needs Hassan's travel time) | |
| TC22 | Routing | Invalid SCATS number | `ValueError` | Kevin | PASS |
| TC23 | Routing | Origin == destination | single-node route, 0 s | Kevin | PASS |
| TC24 | End-to-end | 2000 -> 3002, real graph | 1-5 routes | Kevin | |
| TC25 | End-to-end | Trained model, 8am vs 3am | peak not faster than night | All | |
| TC26 | Search | Adapter heuristic on a 3x3 grid | h(n) <= real cheapest time from every node (A* stays optimal) | Kevin | PASS |
| TC27 | Routing | Road names for the GUI | consecutive same-road segments merged, minutes and km summed | Kevin | PASS |
| TC28 | Search | Yen vs brute force on a 3x3 grid | top-5 costs equal the 5 cheapest of ALL simple paths | Kevin | PASS |

### Extra ideas (add as TC26+ if time allows)
- GUI manual tests (screenshots in the report): invalid input, k = 1, different models give different routes
- Same O/D at 5 different times of day -> travel time curve (nice figure for Insights)
- Every pair of sites -> no crash, timing per query (performance)
