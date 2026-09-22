#!/usr/bin/env python3
"""Optional OOF-only blend optimization; never use hidden-test labels."""
from __future__ import annotations
import argparse
def main():
 p=argparse.ArgumentParser(); p.add_argument('--oof',required=True); a=p.parse_args()
 import pandas as pd
 from teknofest_genomics.models.ensemble import optimize_weights
 d=pd.read_csv(a.oof); print(optimize_weights([d.xgb.to_numpy(),d.lgb.to_numpy(),d.cat.to_numpy()],d.label.to_numpy(),d.panel.to_numpy()))
if __name__=='__main__': main()
