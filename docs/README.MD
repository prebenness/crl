# Docs

This directory is the canonical home for project notes, experiment logs, working status, and planning.

## Layout

- `docs/0_session_logs/`
  - chronological records of major work sessions
  - filename convention: `YYYYMMDD-HHMMSS-short-name.MD`
- `docs/1_closed_topics/`
  - stable reference material and settled explanations
  - use this for docs that should be read repeatedly, not rewritten every session
- `docs/2_active_discussions/`
  - current state of open technical topics
  - use this for "what is true right now", open problems, and next likely moves
- `docs/3_future_roadmap/`
  - backlog and forward-looking plans
  - use this for deferred engineering work and longer-term research goals

## Usage Rules

- At the end of every major piece of work, add a file to `docs/0_session_logs/`.
- If a note becomes stable reference material, move or rewrite it into `docs/1_closed_topics/`.
- If a topic is still active, keep its current summary in `docs/2_active_discussions/` and let session logs record the detailed history.
- Keep root-level docs minimal. The root should point into this directory instead of accumulating ad hoc note files.

## Recommended Reading Order

1. `docs/2_active_discussions/` for the current frontier
2. `docs/1_closed_topics/` for stable reference context
3. `docs/0_session_logs/` for detailed historical work
4. `docs/3_future_roadmap/` for deferred and planned work
