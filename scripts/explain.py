#!/usr/bin/env python3
"""Create a local SHAP summary plot from a model and user-supplied local data."""
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--input', type=Path, required=True); parser.add_argument('--model', type=Path, required=True); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--max-rows', type=int, default=1000); args = parser.parse_args()
    import joblib, pandas as pd
    from teknofest_genomics.data.preprocessing import FeatureSchema, apply_schema
    from teknofest_genomics.explainability.shap_analysis import weighted_shap_values
    artifact = joblib.load(args.model); frame = pd.read_csv(args.input)
    schema = FeatureSchema(artifact['features'], artifact['categorical_features'], artifact['categories'])
    X = apply_schema(frame, schema)[schema.features].head(args.max_rows)
    values = weighted_shap_values(artifact, X)
    import matplotlib.pyplot as plt, shap
    shap.summary_plot(values, X, show=False); plt.tight_layout(); plt.savefig(args.output, dpi=200, bbox_inches='tight'); plt.close()
if __name__ == '__main__': main()
