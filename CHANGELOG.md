## v0.2
- Model: Ridge(alpha=1.0) + StandardScaler
- Improvement: RMSE 53.85 → 53.77
- Precision/Recall at 70th percentile: P=0.857, R=0.5
- Added CI/CD release automation via GHCR

## v0.1
- Baseline: StandardScaler + LinearRegression
- API: /health, /predict endpoints
- RMSE: 53.85
- Dockerized + CI workflow