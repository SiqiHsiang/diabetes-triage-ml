# src/api.py
import os, json
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .schemas import PredictRequest
from .version import __version__ as api_version

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

app = FastAPI(title="Diabetes Progression API")

# load once on startup
model = joblib.load(MODEL_PATH)
metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": api_version,
        "model_name": metrics.get("model_name", "unknown"),
        "rmse": metrics.get("rmse", None),
    }

@app.post("/predict")
def predict(body: PredictRequest):
    try:
        pred = float(model.predict(body.as_row())[0])
        return {"prediction": pred}
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": "prediction_failed", "detail": {"message": str(e)}},
        )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": {"message": str(exc)}},
    )