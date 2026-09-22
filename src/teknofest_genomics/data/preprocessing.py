"""Leakage-aware tabular preparation shared by training and inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import pandas as pd

ID_COLUMN = "Variant_ID"
LABEL_COLUMN = "Label"

@dataclass(frozen=True)
class FeatureSchema:
    features: list[str]
    categorical_features: list[str]
    categories: dict[str, list[str]]

def select_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain the final feature set convention: remove ablated *_eksik flags."""
    return frame[[column for column in frame.columns if not column.endswith("_eksik")]].copy()

def validate_input(frame: pd.DataFrame, features: Iterable[str] | None = None) -> None:
    if frame.empty:
        raise ValueError("Input contains no rows.")
    if ID_COLUMN not in frame:
        raise ValueError(f"Missing required ID column: {ID_COLUMN}")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError("IDs must be non-null and unique.")
    if features is not None:
        missing = [column for column in features if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} required model features: {missing[:5]}")

def fit_schema(train: pd.DataFrame) -> FeatureSchema:
    validate_input(train)
    if LABEL_COLUMN not in train:
        raise ValueError(f"Training data needs {LABEL_COLUMN}.")
    features = [c for c in train.columns if c not in {ID_COLUMN, LABEL_COLUMN, "__SOURCE_PANEL__"}]
    categorical = list(train[features].select_dtypes(include=["object", "string", "category"]).columns)
    categories: dict[str, list[str]] = {}
    for column in categorical:
        values = train[column].astype("string").fillna("_MISSING_")
        vocabulary = list(pd.Series(values.unique()).dropna())
        if "_UNK_" not in vocabulary:
            vocabulary.append("_UNK_")
        categories[column] = vocabulary
    return FeatureSchema(features, categorical, categories)

def apply_schema(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Map only model categorical features; preserve IDs and all metadata intact."""
    validate_input(frame, schema.features)
    result = frame.copy()
    for column, vocabulary in schema.categories.items():
        values = result[column].astype("string").fillna("_MISSING_")
        values = values.where(values.isin(vocabulary), "_UNK_")
        result[column] = pd.Categorical(values, categories=vocabulary)
    return result

def training_matrix(train: pd.DataFrame, schema: FeatureSchema) -> tuple[pd.DataFrame, pd.Series]:
    prepared = apply_schema(train, schema)
    return prepared[schema.features].copy(), prepared[LABEL_COLUMN].astype(int)
