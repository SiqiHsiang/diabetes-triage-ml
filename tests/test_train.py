import json
import subprocess
import sys
import tempfile
import pathlib

def test_training_produces_artifacts():
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "models"
        subprocess.check_call([sys.executable, "-m", "src.train", "--out_dir", str(out)])
        assert (out / "model.pkl").exists()
        m = json.loads((out / "metrics.json").read_text())
        assert m["rmse"] > 0 and "features" in m