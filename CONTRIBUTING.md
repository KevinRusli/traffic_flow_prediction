# Contributing - how we work

## Setup (once)

```bash
git clone <repo-url>
cd traffic_flow_prediction
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest            # everything should be PASS, XFAIL or SKIPPED, never FAIL
```

## Git workflow (same as 2A)

- Work directly on `main`; there are no other branches.
- Run `git pull` before every `git push` so you pick up teammates' changes.
- Only commit your own files (see the ownership table in the README).
- Clear commit messages with a module prefix: `data: drop duplicate rows for site 4335`,
  `model: add LSTM training loop`, `gui: route cards`.

```bash
# ...work, then...
git add src/models/lstm.py tests/test_models.py
git commit -m "model: add LSTM"
git pull
git push
```

If `git push` is rejected, someone pushed first: run `git pull`, then `git push` again.

**Never add `Co-Authored-By:` or any AI attribution line to a commit message.** GitHub turns it into a
"contributor" on the repo. AI use is recorded in `docs/ai-usage-log.md` instead.

## Rules

- **Only edit files you own.** Shared files (`src/graph/graph.py`, `src/models/base.py`,
  `src/prediction/predictor.py`, `config.yaml`, `docs/INTERFACES.md`) change only after telling everyone affected.
- **Never edit `src/search/a2a/`.** It is our 2A code, kept identical on purpose.
- No magic numbers in code -> put them in `config.yaml`.
- Every public function gets a docstring; comment the "why", not the "what". (Up to -10 marks for messy code.)
- Never commit trained models or `data/processed/` (git-ignored). Commit `results/` figures + CSVs, the report uses them.
- Never copy the tutor's example code from `reference/` into `src/`.
- Run `pytest` before pushing. Your own tests must PASS, and you must not turn someone else's PASS into FAIL.
- Log AI use in `docs/ai-usage-log.md` and every resource you use for the report's Acknowledgements.

## Individual-mark warning

The spec allows up to **-60** for an individual who doesn't give progress updates to the tutor in class,
and up to **-60** for not contributing. Come to class with something to show, and push under your own account.
