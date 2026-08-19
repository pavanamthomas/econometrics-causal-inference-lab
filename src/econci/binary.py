"""Binary response models: logit, probit, marginal effects, and classification.

Predicted probabilities, confusion matrices, and ROC/AUC describe
classification performance. They do not identify a causal effect. Average
marginal effects in this module are ceteris paribus derivatives of the
fitted response probability. They inherit whatever identifying content
(or lack of it) the design supplies.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from statsmodels.discrete.discrete_model import BinaryResultsWrapper


def fit_logit(df: pd.DataFrame, formula: str) -> BinaryResultsWrapper:
    """Maximum-likelihood logit."""
    return smf.logit(formula, data=df).fit(disp=False)


def fit_probit(df: pd.DataFrame, formula: str) -> BinaryResultsWrapper:
    """Maximum-likelihood probit."""
    return smf.probit(formula, data=df).fit(disp=False)


def predicted_probabilities(results: BinaryResultsWrapper) -> np.ndarray:
    """In-sample predicted P(y=1 | x)."""
    return np.asarray(results.predict())


def average_marginal_effects(results: BinaryResultsWrapper) -> pd.DataFrame:
    """Overall average marginal effects from statsmodels.

    These are average derivatives of the fitted probability, not treatment
    effects unless the covariates are assigned in a way that justifies that
    reading (see the simulated binary DGP in dgp.simulate_binary_choice).
    """
    me = results.get_margeff(at="overall")
    frame = me.summary_frame()
    frame = frame.reset_index().rename(columns={"index": "variable"})
    return frame


def confusion_at_threshold(
    y: np.ndarray | pd.Series,
    prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Confusion matrix at a probability cutoff used only for classification."""
    y_arr = np.asarray(y).astype(int)
    p_arr = np.asarray(prob, dtype=float)
    yhat = (p_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, yhat, labels=[0, 1]).ravel()
    n = float(len(y_arr))
    return {
        "threshold": float(threshold),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "accuracy": float((tn + tp) / n),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
    }


def roc_auc(y: np.ndarray | pd.Series, prob: np.ndarray | pd.Series) -> dict[str, np.ndarray | float]:
    """ROC curve and AUC. Classification metric only."""
    y_arr = np.asarray(y).astype(int)
    p_arr = np.asarray(prob, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_arr, p_arr)
    auc = float(roc_auc_score(y_arr, p_arr))
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}


def calibration_table(
    y: np.ndarray | pd.Series,
    prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Equal-width bins of predicted probability versus observed frequency.

    A classifier can have a high AUC and still be poorly calibrated. Neither
    quantity answers a causal question by itself.
    """
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_arr, bins, right=True) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            rows.append(
                {
                    "bin": b,
                    "p_low": float(bins[b]),
                    "p_high": float(bins[b + 1]),
                    "n": 0,
                    "mean_predicted": np.nan,
                    "mean_observed": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin": b,
                "p_low": float(bins[b]),
                "p_high": float(bins[b + 1]),
                "n": int(mask.sum()),
                "mean_predicted": float(p_arr[mask].mean()),
                "mean_observed": float(y_arr[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


CLASSIFICATION_VERSUS_CAUSAL = (
    "ROC/AUC, accuracy, and the confusion matrix evaluate assignment of labels "
    "from fitted probabilities. Average marginal effects describe how the fitted "
    "P(y=1|x) changes with x. Causal interpretation of those derivatives requires "
    "a separate argument about how x is assigned. High AUC does not imply that "
    "a coefficient is a treatment effect."
)
