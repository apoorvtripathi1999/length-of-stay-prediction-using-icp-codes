# Hospital Length-of-Stay Prediction

This project predicts a patient's hospital length of stay (LOS), in days, from information available around admission. It combines healthcare-claims data, feature engineering, an XGBoost regression model, and a lightweight web service so that an administrator can obtain an estimate before the bed plan is finalized.

## Objective

The objective is to provide an early, data-informed estimate of inpatient days. The forecast is intended to support:

- bed and capacity planning;
- staffing and care coordination;
- identifying admissions that may need closer operational review; and
- consistent, repeatable estimation instead of relying only on a single population average.

This is an operational planning aid, not a clinical decision, diagnosis, or substitute for a physician's judgment.

## Dataset and target

The data preparation notebooks retrieve New York City hospital-discharge records from the [New York State Health Data API](https://health.data.ny.gov/resource/gnzp-ekau.json), filter for the New York City health service area, and save the source data under `data/`. The target is `length_of_stay`, measured in days.

The transformation workflow derives model-ready features from admission and clinical classification fields, including:

- DRG and major diagnostic category (MDC);
- age group, gender, and ethnicity;
- admission type and emergency-department indicator;
- severity of illness and risk of mortality;
- procedure category and facility; and
- the service-intensity weight.

Categorical variables are one-hot encoded. The deployed artifact stores the resulting feature order in [`artifacts/feature_columns.json`](./artifacts/feature_columns.json), which keeps inference aligned with training.

## Baseline and model performance

### Base score

The base score is a **naive median-prediction baseline with an MAE of 4.42 days**. It is calculated as follows:

1. Calculate the median `length_of_stay` in the training set.
2. Predict that same median for every training example.
3. Calculate mean absolute error:

   `MAE = mean(abs(actual_length_of_stay - predicted_length_of_stay))`

The median is a suitable simple baseline for LOS because the target is right-skewed: a small number of long admissions can pull the mean upward. The median is also a robust “typical stay” estimate. This baseline answers the important question, “Does the feature-based model do better than always predicting a typical stay?”

### XGBoost results

The deployed XGBoost regressor was evaluated on the held-out validation split:

| Metric | XGBoost validation result | Interpretation |
| --- | ---: | --- |
| Mean absolute error (MAE) | **2.9738 days** | Average absolute forecast error |
| Mean squared error (MSE) | **35.9883 days²** | Penalizes larger errors more heavily |
| R² | **0.5939** | Approximately 59.4% of validation-set target variance explained |

The validation MAE improves on the 4.42-day median baseline by approximately **1.45 days**, or **32.7%**:

`(4.42 - 2.9738) / 4.42 * 100 = 32.7%`

That is a meaningful improvement because the model is not merely reproducing the typical stay; it uses patient and admission characteristics to reduce average error by nearly one-third. An MAE of about three days should still be interpreted in context: LOS is variable, the target is skewed, and the model is intended for planning rather than exact discharge-date prediction.

For comparison, the experiment notebook also records the final XGBoost run on the held-out test split:

- MAE: **2.9442 days**
- MSE: **34.7070 days²**
- R²: **0.6084**

The application displays the validation results because those are the metrics associated with the deployed model's validation check. The test results are reported separately to avoid mixing evaluation splits.

## Service architecture

```text
                         Browser
                            |
                            | GET /
                            v
                 +----------------------+
                 | FastAPI + Uvicorn   |
                 | app.py               |
                 +----------+-----------+
                            |
                 POST /predict or /predict-flat
                            |
                            v
                 +----------------------+
                 | Feature alignment    |
                 | pandas DataFrame     |
                 | Missing fields -> 0  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | XGBRegressor         |
                 | artifacts/los_xgb.json|
                 +----------+-----------+
                            |
                            v
              {"predicted_length_of_stay_days": ...}

  index.html + static/app.js + static/styles.css
       provide the responsive form and call the API

  Dockerfile
       builds a Python 3.12 image and runs Uvicorn on port 8000
```

### Request flow

1. The browser serves the admission form from `index.html`.
2. `static/app.js` converts the selected values into the one-hot feature names used during training.
3. The API creates a one-row pandas DataFrame in the exact order from `feature_columns.json`. Fields not supplied by the user are represented as zero.
4. The XGBoost model loaded from `artifacts/los_xgb.json` returns a numeric LOS estimate.
5. The UI displays the result in days and exposes the validation metrics used for the deployed model.

The service also provides:

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Web UI |
| `GET /health` | Basic health check |
| `POST /predict` | Prediction from `{ "features": { ... } }` |
| `POST /predict-flat` | Prediction from either a wrapped or flat feature payload |

The interactive UI accepts partial information. It rejects an entirely empty request, while unspecified individual fields are treated as unknown and left at their zero-encoded value.

## Run locally

### Python

Use Python 3.12, create a virtual environment, and install the serving dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-serving.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>.

### Docker

Build and run the complete service:

```bash
docker build -t length-of-stay-predictor .
docker run --rm -p 8000:8000 length-of-stay-predictor
```

The container copies only the serving requirements, API, frontend assets, and trained artifacts. The training notebooks and development dependencies are not needed at inference time.

Example health check:

```bash
curl http://localhost:8000/health
```

Example prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"service_intensity_weight":1.25,"DRG_Medical":1,"age_group_50 to 69":1,"gender_F":1}}'
```

## Training and experiment tracking

The notebooks document data acquisition, transformations, train/validation/test preparation, model comparison, and evaluation:

- [`get-data.ipynb`](./get-data.ipynb) downloads the source records.
- [`transformers.ipynb`](./transformers.ipynb) prepares and encodes features.
- [`experiments.ipynb`](./experiments.ipynb) evaluates the baseline and regression models.

MLflow runs are stored in the local SQLite tracking database. To inspect them:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Run notebooks from the repository root so their relative data paths and MLflow URI resolve correctly.

## Automated retraining with Airflow

The production retraining workflow is defined in [`dags/los_retraining_dag.py`](./dags/los_retraining_dag.py). It runs weekly on Sunday at 02:00 and executes these guarded steps:

```text
download source data
  -> prepare features and deterministic train/validation/test splits
  -> train candidate XGBoost model
  -> evaluate candidate against the baseline and current validation thresholds
  -> promote only a passing candidate
  -> rebuild the Docker service
  -> verify GET /health
```

The candidate must improve on the median baseline and meet the configured MAE and R² thresholds. A failed evaluation stops the DAG before model promotion or deployment. Each run writes temporary outputs under `runs/<run_id>/`, while production artifacts remain in `artifacts/`.

For a local Airflow installation:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
airflow db migrate
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
airflow dags list --local
airflow scheduler
```

Set the source API credential as an Airflow Variable rather than committing it:

```bash
airflow variables set ny_data_api_key "$NY_DATA_API_KEY"
```

The DAG uses `docker compose up -d --build los-api` for deployment. The Airflow worker therefore needs access to the Docker daemon and the repository directory. For a production setup, replace direct local Docker deployment with an image registry and immutable image tags.

The reusable commands behind the DAG can also be run independently:

```bash
python -m ml.prepare_features --input data/main.csv --output-dir runs/manual/prepared --assets-dir data
python -m ml.train --input-dir runs/manual/prepared --output-dir runs/manual/candidate
python -m ml.evaluate --data-dir runs/manual/prepared --model-dir runs/manual/candidate --output runs/manual/evaluation.json
python -m ml.promote_model --candidate-dir runs/manual/candidate --evaluation runs/manual/evaluation.json --production-dir artifacts
```

## Repository layout

```text
.
├── app.py                         # FastAPI service and inference logic
├── index.html                     # Prediction UI
├── static/                        # Browser JavaScript and CSS
├── artifacts/
│   ├── los_xgb.json               # Trained XGBoost model
│   └── feature_columns.json       # Inference feature order
├── data/                          # Local source and split data
├── get-data.ipynb                 # Data acquisition
├── transformers.ipynb             # Feature engineering
├── experiments.ipynb              # Experiments and evaluation
├── ml/                             # Reusable retraining scripts
├── dags/                           # Airflow orchestration
├── Dockerfile                     # Container image definition
├── docker-compose.yml              # Local API deployment
└── requirements-serving.txt       # Minimal runtime dependencies
```

## Limitations and next steps

- The reported metrics are tied to the available data split and should be re-evaluated when the source data, population, or feature definitions change.
- A point estimate does not communicate uncertainty. Prediction intervals and calibration by LOS range would make the tool more useful for capacity planning.
- The service currently accepts a controlled set of encoded fields; production use would need schema validation, authentication, audit logging, monitoring, and a model/data refresh process.
- Healthcare data can contain institutional and temporal bias. Performance should be checked across facilities, demographic groups, admission types, and time periods before operational deployment.