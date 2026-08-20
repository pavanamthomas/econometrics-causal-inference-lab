"""Wild cluster bootstrap versus cluster-robust Wald with few treated clusters.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import numpy as np

from econci import dgp, panel


def test_wild_cluster_is_not_the_wald_interval() -> None:
    df = dgp.simulate_few_treated_clusters(
        n_clusters=8, n_per_cluster=12, n_treated_clusters=2, beta=1.0, seed=42
    )
    out = panel.wild_cluster_rademacher_interval(
        df, "y", ["treated"], "entity", "treated", n_boot=99, seed=42
    )
    assert out["n_clusters"] == 8
    assert out["n_treated_clusters"] == 2
    assert out["wcb_lower"] < out["wcb_upper"]
    assert (out["wcb_lower"], out["wcb_upper"]) != (out["wald_lower"], out["wald_upper"])
    assert float(np.std(out["replicates"])) > 0.0


def test_row_level_rademacher_is_not_the_cluster_procedure() -> None:
    df = dgp.simulate_few_treated_clusters(
        n_clusters=8, n_per_cluster=12, n_treated_clusters=2, beta=1.0, seed=7
    )
    clustered = panel.wild_cluster_rademacher_interval(
        df, "y", ["treated"], "entity", "treated", n_boot=79, seed=11
    )
    rows = df.copy()
    rows["row_id"] = np.arange(len(rows))
    iid_like = panel.wild_cluster_rademacher_interval(
        rows, "y", ["treated"], "row_id", "treated", n_boot=79, seed=11
    )
    assert clustered["n_clusters"] == 8
    assert iid_like["n_clusters"] == len(rows)
    assert abs(clustered["wcb_upper"] - iid_like["wcb_upper"]) > 1e-8


def test_cluster_wald_undercovers_with_few_treated_clusters() -> None:
    n_reps = 40
    hits = 0
    rng_seeds = range(40, 40 + n_reps)
    for seed in rng_seeds:
        df = dgp.simulate_few_treated_clusters(
            n_clusters=6,
            n_per_cluster=10,
            n_treated_clusters=2,
            beta=1.0,
            sigma_a=3.0,
            seed=int(seed),
        )
        out = panel.wild_cluster_rademacher_interval(
            df, "y", ["treated"], "entity", "treated", n_boot=59, seed=int(seed) + 1000
        )
        if out["wald_lower"] <= 1.0 <= out["wald_upper"]:
            hits += 1
    coverage = hits / n_reps
    # Monte Carlo SE of a binomial proportion at 40 draws is about 0.08.
    # The designed failure is undercoverage of the cluster Wald interval.
    assert coverage < 0.85
