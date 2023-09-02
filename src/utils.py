import os
import sys
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
sys.path.insert(0,'D:\WaferFaultDetection\src')
from logger import *
from exception import *
def save_objects(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        reports={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            Accuracy_score=accuracy_score(y_test,y_pred)
            reports[list(models.keys())[i]] =  Accuracy_score
        return reports
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)


