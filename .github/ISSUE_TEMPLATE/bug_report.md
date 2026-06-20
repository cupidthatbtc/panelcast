---
name: Bug report
about: Report a problem so it can be reproduced and fixed
title: "[bug] "
labels: bug
assignees: ""
---

## What happened

A clear description of the bug.

## What you expected

What you expected to happen instead.

## How to reproduce

The exact command(s) you ran, e.g.:

```bash
panelcast run --dataset aero --num-chains 1 --num-samples 300
```

- Dataset: built-in AOTY defaults / a custom descriptor (please attach or paste
  the relevant fields)
- Stage(s) involved (prepare / train / evaluate / predict / report), if known

## Output

The traceback, error message, or the relevant diagnostics
(R-hat / ESS / PPC / calibration). Trim to the relevant part.

## Environment

- panelcast version (`panelcast --version`):
- OS:
- Python version:
- Install method (pixi / pip):
- GPU (model + VRAM) or CPU-only:

## Additional context

Anything else that might help — recent changes, whether it reproduces on the
AOTY defaults, etc.
