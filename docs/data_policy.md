# Data policy

Copyright 2026 Dr. Pavanam Thomas

## What this repository contains

No observational microdata, administrative extracts, survey files, or scraped
web panels are stored here. The `data/` directory exists only to state that
fact.

All estimation samples are generated at runtime by `src/econci/dgp.py` using
`numpy.random.Generator`. The default seed is 42. Callers may pass another
seed; they should document it.

## What outputs are

Files written to `outputs/tables` and `outputs/figures` by
`python scripts/run_all.py` are finite-sample computational artifacts of
those simulations. They are not estimates from real households, firms,
jurisdictions, or programs. They must not be cited as empirical evidence.

## What causal language is allowed

Causal identification is a property of a design and its assumptions, not of
a Python function. Simulated DGPs can make an assumption true by
construction (parallel trends, exclusion, unconfoundedness, a sharp cutoff).
That is a teaching and validation device. It is not evidence that the same
assumption holds in an application.

## Privacy and licensing

Because there are no personal records, there is no PII workflow. Source code
is MIT-licensed (see `LICENSE`). Copyright 2026 Dr. Pavanam Thomas.

## Reproducibility

Install the package, run the test suite, and regenerate artifacts:

```
pip install -e ".[dev]"
pytest -q
python scripts/run_all.py
```

Do not commit local virtual environments, caches, or editor directories.
See `.gitignore`.
