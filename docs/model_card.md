# Model card

## Intended use

This research prototype ranks tabular genomic variants as benign (`0`) or pathogenic (`1`) using a model score and a fixed 0.50 binary decision threshold. It is intended for reproducible engineering research and variant-prioritization exploration only.

## Model and training

The final ensemble blends XGBoost, LightGBM and CatBoost probabilities, with CatBoost receiving the largest blend weight. Final fitting uses all available labeled training rows with class-balanced sample weighting. It does not train on a synthetically forced 80/20 class distribution.

## Validation scope

Five-fold stratified OOF predictions, LOPO evaluation where applicable, and a panel-aware 80/20 OOF jury-simulation informed validation-side blend analysis. The 80/20 procedure was not a training dataset. The public repository ships no data, trained model, hidden test file, or claimed leaderboard result.

## Limitations and safety

This is not a clinically validated medical device and does not provide a diagnosis, calibrated clinical risk probability, treatment recommendation, or replacement for expert review. Generalization, calibration, and clinical utility require appropriate prospective and external validation.

## Data and privacy

The competition data is private and intentionally excluded. Runtime users must supply authorized local data that conforms to the documented schema.
