# Length of Stay Regression based on the ICD-10 Codes from Healthcare Claims data

## Setup

1. Install Python 3.12
2. Set up a virtual environment with Python 3.12 and add the `requirements.txt` file dependencies.

## View experiment runs

The experiment notebook stores MLflow runs in the repository's `mlflow.db` SQLite database. Run the notebook from the repository root so its data paths and tracking database resolve correctly, then start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the URL printed by MLflow, usually `http://127.0.0.1:5000`. Runs without an explicit experiment name, such as the baseline and cross-validation runs, appear under the `Default` experiment.
