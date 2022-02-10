from typing import Optional, Tuple, Union

import cuml
import numpy as np
import pandas as pd
from xfeat.types import XDataFrame, XSeries

from .base import BaseModel

NNModel = Union[cuml.ElasticNet]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class ElasticNet(BaseModel):
    config = dict()

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs,
    ) -> Tuple[NNModel, dict]:
        model_params = config["model"]["model_params"]

        categorical_cols = config["categorical_cols"]
        self.config["categorical_cols"] = categorical_cols

        for col in categorical_cols:
            if x_train[col].dtype.name == "category":
                x_train[col] = x_train[col].cat.codes
                x_valid[col] = x_valid[col].cat.codes

        mode = config["model"]["mode"]
        self.mode = mode

        self.num_feats = len(x_train.columns)

        model = cuml.ElasticNet(**model_params)

        model.fit(x_train.values, y_train)
        best_score = {"valid_score": model.score(x_valid.values, y_valid)}

        return model, best_score

    def predict(
        self, model: NNModel, features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        for col in self.config["categorical_cols"]:
            if features[col].dtype.name == "category":
                features[col] = features[col].cat.codes

        return model.predict(
            features.values,
        )

    def get_best_iteration(self, model: NNModel) -> int:
        return 0

    def get_feature_importance(self, model: NNModel) -> np.ndarray:
        return np.zeros(self.num_feats)
