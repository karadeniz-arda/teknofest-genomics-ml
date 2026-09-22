# Experimental history

The original working directory contains a large experimental record. The active code retains only the methodology supported by the final validation artifacts.

| Experiment | Evidence-based outcome | Decision |
|---|---|---|
| SMOTENC / synthetic balancing | Historical scripts and final reporting note validation degradation. | Rejected; use balanced sample weighting. |
| Missingness-indicator columns | Final scripts remove columns ending in `_eksik`. | Excluded from final feature set. |
| Learned decision thresholds | It improved one stress proxy slightly but reduced full-LOPO robustness in the recorded comparison. | Fixed 0.50 retained. |
| Base-model hyperparameter tuning | The authoritative confirmation recorded lower 80/20 and full-LOPO scores than manual parameters. | Manual parameters retained. |
| Probability-level blend | Validated with OOF predictions and panel-aware resampling. | Retained. |

No public claim is made about hidden-test or leaderboard performance.
