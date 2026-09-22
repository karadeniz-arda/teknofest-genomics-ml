#!/usr/bin/env python3
"""Train the frozen manual-parameter ensemble on private local panel CSV files."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--data-dir',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True); args=parser.parse_args()
    import joblib, numpy as np, pandas as pd
    import xgboost as xgb, lightgbm as lgb, catboost as cb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_sample_weight
    from teknofest_genomics.data.preprocessing import fit_schema, select_model_columns, training_matrix
    from teknofest_genomics.models.ensemble import SEED, THRESHOLD, manual_parameters, optimize_weights
    args.output_dir.mkdir(parents=True,exist_ok=True)
    names={"MASTER":"master_leakage_free.csv","KANSER":"kanser_flagged.csv","PAH":"pah_flagged.csv","CFTR":"cftr_flagged.csv"}
    panels={name:select_model_columns(pd.read_csv(args.data_dir/file)) for name,file in names.items()}
    common=list(panels['MASTER'].columns)
    if any(list(frame.columns)!=common for frame in panels.values()): raise ValueError('Panel schemas differ after final column selection.')
    train=pd.concat([frame.assign(__SOURCE_PANEL__=name) for name,frame in panels.items()],ignore_index=True)
    schema=fit_schema(train); X,y=training_matrix(train,schema); source=train['__SOURCE_PANEL__'].to_numpy(); y_array=y.to_numpy(); sample_weight=compute_sample_weight(class_weight='balanced',y=y_array)
    params=manual_parameters(schema.categorical_features); fit_params={**params,'cat':{**params['cat'],'train_dir':str(args.output_dir/'catboost_info')}}; oof=[np.zeros(len(X)) for _ in range(3)]
    for train_index,validation_index in StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y_array):
        models=[xgb.XGBClassifier(**fit_params['xgb']).fit(X.iloc[train_index],y_array[train_index],sample_weight=sample_weight[train_index],verbose=False),lgb.LGBMClassifier(**fit_params['lgb']).fit(X.iloc[train_index],y_array[train_index],sample_weight=sample_weight[train_index]),cb.CatBoostClassifier(**fit_params['cat']).fit(X.iloc[train_index],y_array[train_index],sample_weight=sample_weight[train_index])]
        for values,model in zip(oof,models): values[validation_index]=model.predict_proba(X.iloc[validation_index])[:,1]
    # 80/20 groups are used only on OOF predictions to select blend weights.
    # Final base models below are fit on every labeled row with balanced weights.
    weights=optimize_weights(oof,y_array,source)
    final=[xgb.XGBClassifier(**fit_params['xgb']).fit(X,y_array,sample_weight=sample_weight,verbose=False),lgb.LGBMClassifier(**fit_params['lgb']).fit(X,y_array,sample_weight=sample_weight),cb.CatBoostClassifier(**fit_params['cat']).fit(X,y_array,sample_weight=sample_weight)]
    artifact={'xgb_model':final[0],'lgb_model':final[1],'cat_model':final[2],'weights':weights,'threshold':THRESHOLD,'features':schema.features,'categorical_features':schema.categorical_features,'categories':schema.categories,'base_params':params,'architecture':'manual_three_model_probability_blend','training_rows':len(X),'source_panels':list(names)}
    joblib.dump(artifact,args.output_dir/'model.joblib'); (args.output_dir/'manifest.json').write_text(json.dumps({k:v for k,v in artifact.items() if k not in {'xgb_model','lgb_model','cat_model'}},indent=2,default=str)); print(args.output_dir/'model.joblib')
if __name__=='__main__': main()
