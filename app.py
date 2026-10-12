from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "los_xgb.json"
FEATURE_COLUMNS_PATH = BASE_DIR / "artifacts" / "feature_columns.json"

with FEATURE_COLUMNS_PATH.open() as file:
    FEATURE_COLUMNS = json.load(file)

MODEL = xgb.XGBRegressor()
MODEL.load_model(MODEL_PATH)

app = FastAPI(title="Length of Stay Predictor")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ..., 
        description="Feature values matching the training data, for example service_intensity_weight, DRG_Medical, age_group_30 to 49, etc.",
    )


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"message": "Length of Stay Prediction API is running."}


def _to_feature_frame(raw_features: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(raw_features, dict):
        raise ValueError("Features must be a JSON object.")
    if not any(str(value).strip() for value in raw_features.values() if value is not None):
        raise ValueError("At least one feature value is required.")

    feature_row = {column: raw_features.get(column, 0) for column in FEATURE_COLUMNS}
    frame = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    return frame


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    try:
        feature_frame = _to_feature_frame(request.features)
        prediction = float(MODEL.predict(feature_frame)[0])
        return {"predicted_length_of_stay_days": prediction}
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict-flat")
def predict_flat(payload: dict[str, Any]) -> dict[str, float]:
    feature_dict = payload.get("features", payload)
    try:
        feature_frame = _to_feature_frame(feature_dict)
        prediction = float(MODEL.predict(feature_frame)[0])
        return {"predicted_length_of_stay_days": prediction}
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.exception_handler(Exception)
async def handle_exception(_: Any, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
