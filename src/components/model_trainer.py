from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0,'D:\WaferFaultDetection\src')
from logger import *
from exception import *
from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainingConfig:
    model_training_path=os.path.join('artifacts','model.pkl')

class Model_trainer:
    def __init__(self):
        self.model_training_config=ModelTrainingConfig()

    def initiate_model_training(self,train_array,test_array):
        try:

            X_train,y_train,X_test,y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models={
                   'LogisticRegression':LogisticRegression(),
                   'KNN':KNeighborsClassifier(),
                   'RandomForestClassifier':RandomForestClassifier(),
                   'DecisionTreeClassifier':DecisionTreeClassifier(),
                   }
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)
            print(model_report)
            logging.info(f'model report {model_report}')
            best_model_score=max(list(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            print(f'best model found, best model is {best_model_name} with accuracy {best_model_score}')
            logging.info(f'best model found, best model is {best_model_name} with accuracy {best_model_score}')
            save_objects(self.model_training_config.model_training_path,best_model)
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
          
