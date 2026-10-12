from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost LOS model.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    x_train = pd.read_csv(args.input_dir / "x_train.csv")
    y_train = pd.read_csv(args.input_dir / "y_train.csv")["length_of_stay"]
    model = xgb.XGBRegressor(
        n_estimators=1000,
        random_state=42,
        learning_rate=0.20,
        max_depth=8,
        gamma=1,
        reg_alpha=1,
        objective="reg:squarederror",
    )
    model.fit(x_train, y_train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output_dir / "los_xgb.json")
    columns = list(x_train.columns)
    (args.output_dir / "feature_columns.json").write_text(json.dumps(columns, indent=2) + "\n")
    print(f"Saved candidate model with {len(columns)} features to {args.output_dir}")


if __name__ == "__main__":
    main()
