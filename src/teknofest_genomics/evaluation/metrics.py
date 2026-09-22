"""Evaluation metrics used in the competition validation protocol."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score

def evaluate_binary(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.50) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    f1 = f1_score(y_true, prediction, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_true, prediction)
    return {"f1": float(f1), "mcc": float(mcc), "official_score": float((f1 + mcc) / 2),
            "precision": float(precision_score(y_true, prediction, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, probability)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
