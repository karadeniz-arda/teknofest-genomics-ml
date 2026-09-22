# Methodology

The public implementation represents the final validated research direction: a binary (benign/pathogenic) tabular classifier using a CatBoost-dominant XGBoost, LightGBM and CatBoost probability-level blend. It is not a stacking meta-classifier.

## Final training

All available labeled rows from the available panels are used to fit the final three base models. Class imbalance is handled with class-balanced sample weighting; no synthetic or forced 80/20 training distribution is used.

## Validation and jury simulation

Validation used class-balanced sample weighting, five-fold stratified out-of-fold predictions, and leave-one-panel-out evaluation where applicable. Blend weights were optimized only from training-side OOF predictions. A panel-aware 80/20 benign/pathogenic resampling procedure was a validation-side jury simulation used to analyze candidate blend/threshold behavior; it was never the final training corpus. The objective averaged `(pathogenic F1 + MCC) / 2` across source panels. The deployment threshold remained fixed at 0.50 because learned variants were less robust under the recorded validation regimes.

Categorical missing values are represented as `_MISSING_`; categories absent from training are mapped to `_UNK_`. Numeric missing values are left to the tree-model handling. Variant IDs and panel metadata are never model features.
