import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import  AdaBoostRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from dataclasses import dataclass
from src.utils import save_obj
from src.logger import logging
from src.exception import CustomException
from src.utils import Evaluate_model
from src.utils import save_obj

@dataclass
class ModelTrainerConfig:
    model_trainer_config_path = os.path.join('artifacts','modeltrainer.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("In Initiate")
            x_train,y_train,x_test,y_test=(train_arr[:,:-1],
                                           train_arr[:,-1],
                                           test_arr[:,:-1],
                                           test_arr[:,-1])
            print(x_train.shape)
            models = {
                "RR": RandomForestRegressor(),
                "ADA": AdaBoostRegressor(),
                "Ridge":Ridge(),
                "XGBoost":XGBRegressor(),
                "GradBoost":GradientBoostingRegressor()
                    }
            result = Evaluate_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,models=models)
            logging.info("Evalution Done")
            print(result)
            save_obj(file_path=self.model_trainer_config.model_trainer_config_path,
                     obj=result)

        except Exception as e:
            raise CustomException(e,sys)

