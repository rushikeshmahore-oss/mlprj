import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                "CatBoost": CatBoostClassifier(verbose=0),
                "AdaBoost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            print("Model report:")
            print(model_report)

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            threshold = 0.6
            if best_model_score < threshold:
                print(f"All models performed below threshold of {threshold}")
                print("Model scores:", model_report)
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found: {best_model_name} with accuracy: {best_model_score}")
            print(f"✅ Best model: {best_model_name} | Accuracy: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
