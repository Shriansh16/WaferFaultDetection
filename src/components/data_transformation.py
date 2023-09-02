import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
#from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
sys.path.insert(0,'D:\WaferFaultDetection\src')
from logger import *
from exception import *
from utils import *
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
   
    def get_data_transformation_object(self):
        try:
            step1=KNNImputer(n_neighbors=3)
            step2=RobustScaler()
            preprocess_pipe=Pipeline([('step1',step1),('step2',step2)])
            return preprocess_pipe
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            preprocessor=self.get_data_transformation_object()
            target_column_name='Good/Bad'
            
            # training dataset
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            #target_feature_train_df=train_df[target_column_name].map(target_column_mapping)
            #test dataset
            input_feature_test_df=test_df.drop(columns=[target_column_name])
            #target_feature_test_df=test_df[target_column_name].map(target_column_mapping)
            target_feature_test_df=test_df[target_column_name]
            target_feature_train_df=target_feature_train_df.fillna(1)
            target_feature_test_df=target_feature_test_df.fillna(1)
            logging.info(f"missing values independent {input_feature_test_df.isnull().sum()}")
            logging.info(f"missing values dependent {target_feature_test_df.isnull().sum()}")
            logging.info(f"unique categories1 {target_feature_test_df.unique()}")
            logging.info(f"unique categories2 {target_feature_train_df.unique()}")
            logging.info(f"unique counts1 {target_feature_test_df.value_counts()}")
            logging.info(f"unique counts2 {target_feature_train_df.value_counts()}")
            transformed_input_train=preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test=preprocessor.transform(input_feature_test_df)
            #resample=SMOTETomek(sampling_strategy='auto',k_neighbors=2)
            #resample=SMOTE(k_neighbors=2,random_state=0)
            #sampler = SMOTE(ratio={1: 1927, 0: 300},random_state=0)
            
            #input_feature_train_final,target_feature_train_final=resample.fit_resample(transformed_input_train,target_feature_train_df)
            #input_feature_test_final,target_feature_test_final=resample.fit_resample(transformed_input_test,target_feature_test_df)
            #input_feature_test_final,target_feature_test_final=sampler.fit_resample(transformed_input_test,target_feature_test_df)
            train_arr=np.c_[transformed_input_train,np.array(target_feature_train_df)]
            test_arr=np.c_[transformed_input_test,np.array(target_feature_test_df)]
            save_objects(self.data_transformation_config.preprocessor_obj_file_path,preprocessor)
            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("ERROR OCCURED IN DATA TRANSFORMATION")
            raise CustomException(e,sys)






    