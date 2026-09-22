#!/usr/bin/env python3
"""Generate generic JSON predictions from a local model artifact."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 import joblib,pandas as pd
 from teknofest_genomics.models.inference import predict
 frame=pd.read_csv(a.input); artifact=joblib.load(a.model)
 if 'Label' in frame: print('Warning: Label column ignored during inference.')
 result=predict(frame,artifact); a.output.write_text(json.dumps({'predictions':[{**row,'predicted_prob':round(float(row['predicted_prob']),8)} for row in result.to_dict('records')]},indent=2,allow_nan=False)); print(a.output)
if __name__=='__main__': main()
