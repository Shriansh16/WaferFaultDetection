import pandas as pd
import os
import sys
from pathlib import Path
sys.path.insert(0,'D:\WaferFaultDetection\src')
from logger import *
from exception import *
from utils import *
from components.data_ingestion import DataIngestion
sys.path.insert(0, 'D:\WaferFaultDetection\src\components')
from data_ingestion import *
from data_transformation import *
from model_trainer import *

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    obj1=DataTranformation()
    train_data_array,test_data_array,_=obj1.initiate_data_transformation(train_data_path,test_data_path)

    obj2=Model_trainer()
    obj2.initiate_model_training(train_data_array,test_data_array)

