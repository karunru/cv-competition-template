from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from xfeat.types import XDataFrame, XSeries

from .base import BaseModel

CatModel = Union[CatBoostClassifier, CatBoostRegressor]
AoD = Union[np.ndarray, XDataFrame]
AoS = Union[np.ndarray, XSeries]


class CatBoost(BaseModel):
    config = dict()

    def fit(
        self,
        x_train: AoD,
        y_train: AoS,
        x_valid: AoD,
        y_valid: AoS,
        config: dict,
        **kwargs,
    ) -> Tuple[CatModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["train_params"]["mode"]
        self.mode = mode

        categorical_cols = config["categorical_cols"]
        self.config["categorical_cols"] = categorical_cols

        if mode == "regression":
            # model = CatBoostRegressor(
            #     cat_features=self.config["categorical_cols"], **model_params
            # )
            model = CatBoostRegressor(**model_params)
        else:
            # model = CatBoostClassifier(
            #     cat_features=self.config["categorical_cols"], **model_params
            # )
            model = CatBoostClassifier(**model_params)

        train_pool = Pool(
            data=x_train, label=y_train, cat_features=self.config["categorical_cols"]
        )
        val_pool = Pool(
            data=x_valid, label=y_valid, cat_features=self.config["categorical_cols"]
        )
        model.fit(
            train_pool,
            # cat_features=self.config["categorical_cols"],
            eval_set=val_pool,
            verbose=model_params["early_stopping_rounds"],
        )
        best_score = model.best_score_
        return model, best_score

    def get_best_iteration(self, model: CatModel) -> int:
        return model.best_iteration_

    def predict(self, model: CatModel, features: AoD) -> np.ndarray:
        # if model.get_param("loss_function")
        test_pool = Pool(data=features, cat_features=self.config["categorical_cols"])
        if self.mode == "binary":
            return model.predict_proba(test_pool)[:, 1]
        else:
            return model.predict(test_pool)

    def get_feature_importance(self, model: CatModel) -> np.ndarray:
        return model.feature_importances_
