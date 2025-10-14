# src/schemas.py
from pydantic import BaseModel

FEATURE_NAMES = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]

class PredictRequest(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    def as_row(self):
        return [[getattr(self, f) for f in FEATURE_NAMES]]