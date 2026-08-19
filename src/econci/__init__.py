"""Econometrics and causal-inference laboratory.

Applied workflow: problem, formalization, assumptions, estimation,
validation, interpretation, limitations.

All samples are simulated from documented data-generating processes.
Causal language is reserved for designs whose identifying variation and
assumptions are stated. Conventional two-way fixed effects is not
automatically valid under staggered adoption with heterogeneous effects.

Copyright 2026 Dr. Pavanam Thomas
"""

from econci import binary, dgp, did, iv, matching, ols, panel, plots, rd

__version__ = "0.1.0"
__author__ = "Dr. Pavanam Thomas"
__license__ = "MIT"

__all__ = [
    "binary",
    "dgp",
    "did",
    "iv",
    "matching",
    "ols",
    "panel",
    "plots",
    "rd",
    "__version__",
]
