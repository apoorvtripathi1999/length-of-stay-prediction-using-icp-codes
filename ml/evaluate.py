from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a candidate LOS model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mae", type=float, default=2.9737913608551025)
    parser.add_argument("--min-r2", type=float, default=0.593927264213562)
    args = parser.parse_args()

    x_valid = pd.read_csv(args.data_dir / "x_valid.csv")
    y_train = pd.read_csv(args.data_dir / "y_train.csv")["length_of_stay"]
    y_valid = pd.read_csv(args.data_dir / "y_valid.csv")["length_of_stay"]
    model = xgb.XGBRegressor()
    model.load_model(args.model_dir / "los_xgb.json")
    predictions = model.predict(x_valid)
    baseline_mae = mean_absolute_error(y_train, [y_train.median()] * len(y_train))
    metrics = {
        "baseline_mae": float(baseline_mae),
        "mae": float(mean_absolute_error(y_valid, predictions)),
        "mse": float(mean_squared_error(y_valid, predictions)),
        "r2": float(r2_score(y_valid, predictions)),
    }
    metrics["passed"] = metrics["mae"] <= args.max_mae and metrics["mae"] < metrics["baseline_mae"] and metrics["r2"] >= args.min_r2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise RuntimeError("Candidate model did not pass the promotion gate.")


if __name__ == "__main__":
    main()
