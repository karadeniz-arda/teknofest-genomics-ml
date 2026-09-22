# Genomic Variant Pathogenicity Classification

Research code for a competition prototype that classifies genomic variants as benign (`0`) or pathogenic (`1`) from mixed tabular features. This public repository contains methodology and reproducible code only; no competition-provided datasets, organizer test files, trained artifacts, submission JSONs, identifiers, or private delivery material are included.

## Overview

The project addresses a small, heterogeneous tabular classification problem with missing values, categorical inputs, class imbalance, and panel distribution shift. The final approach uses XGBoost, LightGBM, and CatBoost as a weighted probability-level ensemble. It is not a stacking model and it is not a clinical diagnostic system.

## Approach

- Remove ablated missingness-indicator columns (`*_eksik`) from the final feature set.
- Preserve numeric missingness for tree-based learners; encode categorical missing values as `_MISSING_` and unseen categories as `_UNK_`.
- Exclude IDs and panel provenance metadata from model features.
- Train the final models on all available labeled rows using balanced sample weights; no synthetic 80/20 training set is created.
- Use five-fold stratified OOF probabilities and a panel-aware 80/20 jury-simulation only for validation-side blend analysis.
- Optimize only blend weights with Optuna; use a fixed 0.50 decision threshold.
- Evaluate with pathogenic F1, MCC, PR-AUC, and the documented internal proxy `(F1 + MCC) / 2`.

The final probability-level blend was CatBoost-dominant, with smaller XGBoost and LightGBM contributions. The archived validation artifacts support retaining manual base-model parameters and a fixed 0.50 threshold rather than aggressive base-parameter tuning or learned thresholds. See [docs/methodology.md](docs/methodology.md) and [docs/experiments.md](docs/experiments.md).

## Data policy

Competition data is deliberately absent. Supply private local data at runtime with the four expected files:

```text
<private-data-dir>/
  master_leakage_free.csv
  kanser_flagged.csv
  pah_flagged.csv
  cftr_flagged.csv
```

Each file must contain `Variant_ID`, `Label`, and the same feature schema. `Label` is binary: `0` benign and `1` pathogenic. [examples/example_schema.csv](examples/example_schema.csv) is synthetic and illustrative only; it is not trainable competition data.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Train

```bash
python scripts/train_final.py \
  --data-dir /path/to/private/data \
  --output-dir artifacts
```

This writes a local `artifacts/model.joblib`, which is ignored by Git.

## Predict

```bash
python scripts/predict.py \
  --input /path/to/input.csv \
  --model artifacts/model.joblib \
  --output predictions.json
```

Inference preserves `Variant_ID`, rejects duplicate/missing IDs and missing required features, ignores any accidental `Label` column, maps unseen categorical model features to `_UNK_`, and emits finite class-1 model scores. The JSON is generic and intentionally contains no competition team metadata.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
python scripts/train_final.py --help
python scripts/predict.py --help
```

## Limitations and ethics

This is a competition/research prototype, not a clinically validated medical device. Results are internal validation measurements on private competition data; they do not establish clinical utility, population generalization, calibration, or diagnostic validity.

## Public-release review

Before publishing, complete [PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md). In particular, review any new artifacts or examples to ensure they contain no organizer-provided data, test IDs, team/application information, credentials, or personal paths.

Competition presentation materials are intentionally excluded from the public repository because they contain third-party event branding.
