# src/train.py
import json, os, random
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

SEED = 42  # for reproducibility

def main(out_dir: str = "models"):

    os.makedirs(out_dir, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED)

    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    meta = {
        "model_name": "StandardScaler+LinearRegression",
        "seed": SEED,
        "rmse": rmse,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X.columns),
    }
    print("Saving model and metrics to:", out_dir)
    joblib.dump(pipe, os.path.join(out_dir, "model.pkl"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="models")
    args = p.parse_args()
    main(args.out_dir)
