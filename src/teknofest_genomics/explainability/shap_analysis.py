"""Minimal model-level SHAP utilities; no data or plots are shipped."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _positive_values(values):
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values

def weighted_shap_values(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Return probability-blend-weighted tree SHAP contributions for local analysis."""
    import shap
    models = [artifact["xgb_model"], artifact["lgb_model"], artifact["cat_model"]]
    weights = [artifact["weights"][name] for name in ("xgb", "lgb", "cat")]
    contributions = [_positive_values(shap.TreeExplainer(model).shap_values(X)) for model in models]
    return sum(weight * value for weight, value in zip(weights, contributions))
