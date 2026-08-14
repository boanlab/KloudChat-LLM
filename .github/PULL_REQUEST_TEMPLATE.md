## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- The problem this solves. Include the failure you saw, if it was a bug. -->

## How it was verified

<!-- Commands you ran and what they printed. "Not verified on hardware" is a fine
     answer for GPU-node changes — say so explicitly rather than leaving it blank. -->

- [ ] `pytest scheduler/tests -q`
- [ ] `bash -n scripts/*.sh` (or shellcheck)
- [ ] `docker compose config --quiet` for any compose change
- [ ] Docs updated for any flag, environment variable, or default that moved
