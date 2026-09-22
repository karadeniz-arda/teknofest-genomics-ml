"""Generic, deterministic inference without competition metadata."""
from __future__ import annotations
import numpy as np
import pandas as pd
from teknofest_genomics.data.preprocessing import FeatureSchema, apply_schema, validate_input

def positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1 in classes: index = classes.index(1)
    elif "1" in classes: index = classes.index("1")
    else: raise ValueError("Model does not expose pathogenic class 1.")
    return model.predict_proba(X)[:, index]

def predict(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    schema = FeatureSchema(artifact["features"], artifact["categorical_features"], artifact["categories"])
    validate_input(frame, schema.features)
    raw_ids = frame["Variant_ID"].astype(str).copy()
    prepared = apply_schema(frame, schema)
    X = prepared[schema.features]
    weights = artifact["weights"]
    score = weights["xgb"] * positive_probability(artifact["xgb_model"], X) + weights["lgb"] * positive_probability(artifact["lgb_model"], X) + weights["cat"] * positive_probability(artifact["cat_model"], X)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any(): raise ValueError("Non-finite or out-of-range prediction.")
    classes = (score >= artifact["threshold"]).astype(int).astype(str)
    out = pd.DataFrame({"id": raw_ids, "predicted_class": classes, "predicted_prob": score})
    if not out["id"].equals(raw_ids) or out["id"].duplicated().any(): raise ValueError("ID preservation check failed.")
    return out
