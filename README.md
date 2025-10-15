Machine learning API for predicting diabetes progression scores based on patient features. Built with FastAPI, scikit-learn, and Docker, with continuous integration and release to GitHub Container Registry (GHCR).

## 1. Overview
Two model versions are available:
#### v0.1.0
Model: StandardScaler + LinearRegression
RMSE: 53.85
Description: Baseline model
#### v0.2.2
Model: StandardScaler + Ridge(alpha=1.0)
RMSE: 53.77
Description: Added regularization and CI/CD release

## 2. Quick Start (run prebuilt image)
#### v0.1 (Linear Regression)
```
docker pull ghcr.io/siqihsiang/diabetes-triage-ml:v0.1
docker run --rm -p 8000:8000 ghcr.io/siqihsiang/diabetes-triage-ml:v0.1
```
#### v0.2 (Ridge)
```
docker pull ghcr.io/siqihsiang/diabetes-triage-ml:v0.2
docker run --rm -p 8080:8000 ghcr.io/siqihsiang/diabetes-triage-ml:v0.2
```
Then test the API:
```
curl http://127.0.0.1:8000/health
```
Example response:
```
{
  "status": "ok",
  "model_version": "0.2.0",
  "model_name": "StandardScaler+Ridge(alpha=1.0)",
  "rmse": 53.77
}
```

## 3. Prediction Endpoint
Sample request
```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001
  }'
```
Sample response
```
{
  "prediction": 182.36
}
```

## 4. Run Locally (retrain or debug)
```
# Train baseline
python -m src.train --out_dir models --kind linear

# Train Ridge model
python -m src.train --out_dir models --kind ridge

# Run API locally
uvicorn src.api:app --reload --port 8000
```

## 5. Project Structure
```
.
├── src/
│   ├── train.py          # model training and evaluation
│   ├── api.py            # FastAPI app
│   ├── model_utils.py    # helper functions
│
├── models/               # saved models and metrics
├── tests/                # pytest unit tests
├── .github/workflows/    # CI/CD pipelines
├── Dockerfile
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## 6. Continuous Integration / Delivery
	•	CI workflow runs lint, unit tests, and smoke-tests the API with pytest.
	•	Release workflow builds and pushes images to GHCR with semantic version tags (v0.1, v0.2, etc.).
	•	Docker images are self-contained and include the trained model.

## 7. Authors
Group AG
