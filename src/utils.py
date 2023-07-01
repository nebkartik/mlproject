#to implement something to be used for the entire app
# for ex - building the code, model to cloud

import os
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def Evaluate_model(x_train,x_test,y_train,y_test,models):
    report = {}

    for model in models.values():
                model.fit(x_train,y_train)
                train_pred = model.predict(x_test)
                test_pred = model.predict(x_test)
                report[model] = r2_score(y_test,test_pred)
    return report