#!/usr/bin/env python3
"""Simple CLI to run `model_inference.predict_sales` inside the container.

Example:
  python -m app.predict_cli --new_csv /data/new_media.csv --out_csv /data/pred.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import pandas as pd

from app.model_inference import predict_sales, _clean_and_format_new_df


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Run MMM posterior predictive on new media spend rows")
    p.add_argument("--model_nc", default=os.getenv("MMM_POSTERIOR_PATH", "artifacts/trained_pymc_mmm_model_01.nc"))
    p.add_argument("--training_csv", default=os.getenv("TRAINING_CSV_PATH", "artifacts/pymc_data.csv"))
    p.add_argument("--new_csv", required=True, help="Path to CSV with new media spend rows (must include 'week' + channels)")
    p.add_argument("--out_csv", default=None, help="If set, write predictions to this CSV; otherwise print to stdout")
    p.add_argument("--hdi_prob", type=float, default=0.9)

    args = p.parse_args(argv)

    if not os.path.exists(args.model_nc):
        raise SystemExit(f"Model .nc not found: {args.model_nc}")
    if not os.path.exists(args.training_csv):
        raise SystemExit(f"Training CSV not found: {args.training_csv}")
    if not os.path.exists(args.new_csv):
        raise SystemExit(f"New CSV not found: {args.new_csv}")

    new_df = pd.read_csv(args.new_csv)
    new_df = _clean_and_format_new_df(new_df, train_df=pd.read_csv(args.training_csv))

    out_df = predict_sales(
        model_nc_path=args.model_nc,
        new_df=new_df,
        training_csv_path=args.training_csv,
        hdi_prob=args.hdi_prob,
    )

    if args.out_csv:
        out_df.to_csv(args.out_csv, index=False)
        print(f"Wrote predictions to {args.out_csv}")
    else:
        print(out_df.to_csv(index=False))


if __name__ == "__main__":
    main()
