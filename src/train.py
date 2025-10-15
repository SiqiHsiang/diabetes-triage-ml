import os
import json
import joblib
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

SEED = 42

def build_model(kind: str = "linear"):
    if kind == "linear":
        name = "StandardScaler+LinearRegression"
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return name, Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])

    if kind == "ridge":
        name = "StandardScaler+Ridge(alpha=1.0)"
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return name, Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=SEED))])

    if kind == "rf":
        name = "RandomForestRegressor(n_estimators=300)"
        return name, RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)

    raise ValueError(f"Unknown kind: {kind}")

def main(out_dir: str = "models", kind: str = "linear"):
    # Load data
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # Build and train
    model_name, model = build_model(kind)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Base metrics
    meta = {
        "model_name": model_name,
        "seed": SEED,
        "rmse": rmse,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X.columns),
    }

    # Optional: flag metrics (top 30% risk)
    threshold = float(np.percentile(y_train, 70))
    y_true_flag = (y_test >= threshold).astype(int)
    y_pred_flag = (y_pred >= threshold).astype(int)
    meta["flag_threshold"] = threshold
    meta["precision_at_thresh"] = float(precision_score(y_true_flag, y_pred_flag))
    meta["recall_at_thresh"] = float(recall_score(y_true_flag, y_pred_flag))

    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="models")
    p.add_argument("--kind", default="linear", choices=["linear","ridge","rf"])
    args = p.parse_args()
    main(args.out_dir, args.kind)