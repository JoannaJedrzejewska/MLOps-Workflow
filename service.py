import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Literal

class PenguinFeatures(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: Literal["Torgersen", "Biscoe", "Dream"]
    sex: Literal["MALE", "FEMALE"]

class PenguinPrediction(BaseModel):
    species: str
    bill_body_ratio: float

@bentoml.service(name="PenguinsService")
class PenguinsService:
    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model("penguins_classifier:latest")
        self.encoder = bentoml.sklearn.load_model("penguins_encoder:latest")

    @bentoml.api
    def classify(self, data: PenguinFeatures) -> PenguinPrediction:
        bill_body_ratio = (
            data.culmen_length_mm * data.culmen_depth_mm
        ) / data.body_mass_g

        cat_df = pd.DataFrame(
            [[data.island, data.sex]],
            columns=["island", "sex"],
        )
        
        X_cat = self.encoder.transform(cat_df) 

        X_num = np.array([[
            data.culmen_length_mm,
            data.culmen_depth_mm,
            data.flipper_length_mm,
            data.body_mass_g,
            bill_body_ratio,
        ]])

        X = np.hstack([X_cat, X_num])
        
        y_pred = self.model.predict(X)

        return PenguinPrediction(
            species=str(y_pred[0]),
            bill_body_ratio=round(bill_body_ratio, 6),
        )