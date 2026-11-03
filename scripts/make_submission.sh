#!/usr/bin/env bash
# Build the Canvas submission zip from the current commit.
# Files marked export-ignore in .gitattributes (reference/, notebooks/, .github/) are left out,
# so the tutor's example scripts are never submitted by accident.
#   usage: scripts/make_submission.sh <TEAM_ID>
set -euo pipefail
TEAM_ID="${1:?usage: scripts/make_submission.sh <TEAM_ID>}"
cd "$(git rev-parse --show-toplevel)"
if [ ! -f "report/${TEAM_ID}.pdf" ] && [ ! -f "report/${TEAM_ID}.docx" ]; then
  echo "WARNING: report/${TEAM_ID}.pdf (or .docx) not found - the report must be named after the Team ID." >&2
fi
git archive --format=zip --prefix="${TEAM_ID}/" -o "${TEAM_ID}.zip" HEAD
echo "Created ${TEAM_ID}.zip"
