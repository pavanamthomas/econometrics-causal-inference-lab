"""Figures for diagnostics and design checks. Matplotlib only.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_residuals_vs_fitted(
    fitted: np.ndarray,
    residual: np.ndarray,
    path: str | Path,
    title: str = "Residuals versus fitted values",
) -> Path:
    out = _prepare(path)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(fitted, residual, s=12, alpha=0.55, c="#1f4e79", linewidths=0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_event_study(coefs: pd.DataFrame, path: str | Path, title: str = "Event-study coefficients") -> Path:
    out = _prepare(path)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    k = coefs["k"].to_numpy()
    b = coefs["coef"].to_numpy()
    se = coefs["se"].to_numpy()
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(-0.5, color="grey", linewidth=0.8, linestyle="--")
    ax.errorbar(k, b, yerr=1.96 * se, fmt="o", color="#1f4e79", ecolor="#1f4e79", capsize=3)
    ax.set_xlabel("Event time (omitted: -1)")
    ax.set_ylabel("Coefficient")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_rd(
    df: pd.DataFrame,
    bins: pd.DataFrame,
    cutoff: float,
    path: str | Path,
    running: str = "running",
    y: str = "y",
    title: str = "Sharp RD: binned means",
) -> Path:
    out = _prepare(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(df[running], df[y], s=6, alpha=0.12, c="grey", linewidths=0)
    for side, sub in bins.groupby("side"):
        ax.plot(sub["mid"], sub["mean"], "o-", color="#1f4e79" if side == "right" else "#9c2a2a", label=side)
    ax.axvline(cutoff, color="black", linewidth=1.0)
    ax.set_xlabel("Running variable")
    ax.set_ylabel("Outcome")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_propensity_overlap(
    ps: np.ndarray,
    treat: np.ndarray,
    path: str | Path,
    title: str = "Propensity-score overlap",
) -> Path:
    out = _prepare(path)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    d = np.asarray(treat).astype(int)
    p = np.asarray(ps, dtype=float)
    ax.hist(p[d == 0], bins=20, alpha=0.55, label="Control", color="#9c2a2a", density=True)
    ax.hist(p[d == 1], bins=20, alpha=0.55, label="Treated", color="#1f4e79", density=True)
    ax.set_xlabel("Fitted propensity score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, path: str | Path) -> Path:
    out = _prepare(path)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(fpr, tpr, color="#1f4e79", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (classification metric, not identification)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_calibration(cal: pd.DataFrame, path: str | Path) -> Path:
    out = _prepare(path)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    m = cal.dropna(subset=["mean_predicted", "mean_observed"])
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.9)
    ax.plot(m["mean_predicted"], m["mean_observed"], "o-", color="#1f4e79")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (classification, not causal)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_staggered_comparison(labels: list[str], values: list[float], path: str | Path) -> Path:
    out = _prepare(path)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(labels, values, color="#1f4e79")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Estimate / known ATT")
    ax.set_title("Staggered design: TWFE versus known cohort ATTs")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
