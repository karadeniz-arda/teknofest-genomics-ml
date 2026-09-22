"""Manual three-model ensemble and OOF blend optimization."""
from __future__ import annotations
import numpy as np
import optuna
from sklearn.metrics import f1_score, matthews_corrcoef

SEED = 42
THRESHOLD = 0.50

def manual_parameters(categorical_features: list[str]) -> dict[str, dict]:
    return {
        "xgb": {"tree_method": "hist", "enable_categorical": True, "random_state": SEED, "n_estimators": 300, "learning_rate": .03, "max_depth": 5, "subsample": .8, "colsample_bytree": .8},
        "lgb": {"random_state": SEED, "n_estimators": 300, "learning_rate": .03, "max_depth": 5, "num_leaves": 31, "subsample": .8, "colsample_bytree": .8, "verbose": -1},
        "cat": {"random_state": SEED, "iterations": 300, "learning_rate": .03, "depth": 5, "l2_leaf_reg": 3.0, "cat_features": categorical_features, "verbose": False},
    }

def panel_aware_groups(y: np.ndarray, panels: np.ndarray, seeds=range(100, 120)) -> list[list[np.ndarray]]:
    """Validation-only 20× 4:1 jury-simulation batches from training-side OOF rows."""
    output = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        groups = []
        for panel in sorted(set(panels)):
            index = np.where(panels == panel)[0]
            benign, pathogenic = index[y[index] == 0], index[y[index] == 1]
            n_pathogenic = min(len(pathogenic), len(benign) // 4)
            if n_pathogenic:
                groups.append(np.concatenate([rng.choice(benign, 4 * n_pathogenic, replace=False), rng.choice(pathogenic, n_pathogenic, replace=False)]))
        output.append(groups)
    return output

def optimize_weights(oof: list[np.ndarray], y: np.ndarray, panels: np.ndarray, trials: int = 100) -> dict[str, float]:
    groups = panel_aware_groups(y, panels)
    def objective(trial: optuna.Trial) -> float:
        raw = np.array([trial.suggest_float("w_xgb", 0, 1), trial.suggest_float("w_lgb", 0, 1), trial.suggest_float("w_cat", 0, 1)])
        if raw.sum() == 0:
            return 0.0
        p = sum(weight * score for weight, score in zip(raw / raw.sum(), oof))
        values = []
        for seed_groups in groups:
            for index in seed_groups:
                pred = (p[index] >= THRESHOLD).astype(int)
                values.append((f1_score(y[index], pred, pos_label=1, zero_division=0) + matthews_corrcoef(y[index], pred)) / 2)
        return float(np.mean(values))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    raw = np.array([study.best_params["w_xgb"], study.best_params["w_lgb"], study.best_params["w_cat"]])
    weights = raw / raw.sum()
    return {"xgb": float(weights[0]), "lgb": float(weights[1]), "cat": float(weights[2])}
