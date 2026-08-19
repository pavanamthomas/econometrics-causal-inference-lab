"""Sharp RD: local estimates versus far-from-cutoff extrapolation.

Copyright 2026 Dr. Pavanam Thomas
"""

from econci import dgp, rd


def test_local_rd_nearer_truth_than_far_extrapolation() -> None:
    df = dgp.simulate_sharp_rd(n=1600, cutoff=0.0, tau=2.0, seed=42)
    local = rd.local_linear_rd(df, cutoff=0.0, bandwidth=0.7, kernel="triangular")
    far = rd.naive_far_difference(df, cutoff=0.0, margin=1.1)
    assert abs(local["tau"] - 2.0) < abs(far["far_diff"] - 2.0)
    assert abs(local["tau"] - 2.0) < 0.35


def test_bandwidth_sensitivity_runs() -> None:
    df = dgp.simulate_sharp_rd(n=800, seed=42)
    table = rd.bandwidth_sensitivity(df, bandwidths=[0.4, 0.8, 1.2], cutoff=0.0)
    assert len(table) == 3
    assert table["tau"].notna().all()
