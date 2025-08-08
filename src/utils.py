import os
import sys 
import numpy as np
import pandas as pd
import dill
import pickle

import pickle
import os
from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score

from src.exception import CustomException
import sys

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    report = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        report[name] = score
    return report


    