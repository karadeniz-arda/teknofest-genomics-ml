#!/usr/bin/env python3
"""Validate final private panel files without copying or modifying them."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from teknofest_genomics.data.preprocessing import select_model_columns, validate_input
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); a=p.parse_args(); frame=select_model_columns(pd.read_csv(a.input)); validate_input(frame); print(f'rows={len(frame)} columns={len(frame.columns)}')
if __name__=='__main__': main()
