import pandas as pd
import numpy as np
import os
import sys
import pickle
import shutil
from flask import request
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, 'D:\WaferFaultDetection\src')
from logger import *
from exception import *
from utils import *

@dataclass
class PredictionPipelineConfig:
    prediction_output_file_directory="predictions"
    prediction_file_name="prediction_file.csv"
    prediction_file_path=os.path.join(prediction_output_file_directory,prediction_file_name)
    model_file_path=os.path.join('artifacts','model.pkl')
    transformer_file_path=os.path.join('artifacts','preprocessor.pkl')

class PredictionPipeline:
    def __init__(self,request:request):
        self.request=request
        self.prediction_pipeline_config=PredictionPipelineConfig()

    def save_input_files(self):
        try:
            pred_file_input_dir="prediction_artifacts"
            os.makedirs(pred_file_input_dir,exist_ok=True)
            input_csv_file=self.request.files['file']
            pred_file_path=os.path.join(pred_file_input_dir,input_csv_file.filename)
            input_csv_file.save(pred_file_path)
            return pred_file_path

        except Exception as e:
            raise CustomException(e,sys)

    def predict(self,features):
        try:
            model=load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor=load_object(self.prediction_pipeline_config.transformer_file_path)
            transformed_fea=preprocessor.fit_transform(features)
            preds=model.predict(transformed_fea)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

    def get_predicted_dataframe(self,input_dataframe_path:pd.DataFrame):
        try:
            prediction_column_name:str= 'TARGET_COLUMN'
            input_dataframe:pd.DataFrame=pd.read_csv(input_dataframe_path)
            input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe
            predictions=self.predict(input_dataframe)
            input_dataframe[prediction_column_name]=[pred for pred in predictions]
            target_column_mapping={0:'bad',1:'good'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            os.makedirs( self.prediction_pipeline_config.prediction_output_file_directory, exist_ok= True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
            logging.info("predictions completed. ")
        except Exception as e:
            raise CustomException(e,sys)
    def run_pipeline(self):
        try:
            input_csv_path=self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)
            return self.prediction_pipeline_config
        except Exception as e:
            raise CustomException(e,sys)
