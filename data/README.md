# Data policy

Copyright 2026 Dr. Pavanam Thomas

This repository does not ship observational, administrative, or survey microdata.

All estimation samples are generated at runtime from documented data-generating processes in `src/econci/dgp.py`. Default simulations use `numpy.random.Generator` with seed 42 unless a caller passes another seed.

Finite-sample tables and figures written to `outputs/` are computational artifacts of those simulations. They are not estimates from real markets, firms, households, or public programs, and they should not be cited as empirical evidence.

See `docs/data_policy.md` for the full statement.
