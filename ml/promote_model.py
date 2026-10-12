from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a validated candidate model.")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--production-dir", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text())
    if not evaluation.get("passed", False):
        raise RuntimeError(f"Refusing to promote failed evaluation: {evaluation}")
    for filename in ("los_xgb.json", "feature_columns.json"):
        if not (args.candidate_dir / filename).is_file():
            raise FileNotFoundError(args.candidate_dir / filename)

    args.production_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix="los-model-", dir=args.production_dir.parent))
    try:
        for filename in ("los_xgb.json", "feature_columns.json"):
            shutil.copy2(args.candidate_dir / filename, temporary_dir / filename)
        args.production_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("los_xgb.json", "feature_columns.json"):
            shutil.copy2(temporary_dir / filename, args.production_dir / filename)
        (args.production_dir / "metrics.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    print(f"Promoted candidate to {args.production_dir}")


if __name__ == "__main__":
    main()
