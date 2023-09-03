import os
import sys
import pandas as pd
sys.path.insert(0,'D:\WaferFaultDetection\src')
from logger import *
from exception import *
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("DATA INGESTION STARTS")

        try:
            df=pd.read_csv(os.path.join('notebooks','cleaned_data_final4.csv'))
            logging.info("dataset read as pandas dataframe")
            logging.info(df.head)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info("TRAIN TEST SPLIT")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path)
            test_set.to_csv(self.data_ingestion_config.test_data_path)
            logging.info("INGESTION OF DATA COMPLETED")

            return(self.data_ingestion_config.train_data_path,
                   self.data_ingestion_config.test_data_path)

        except Exception as e:
            logging.info("ERROR HAS OCCURED IN DATA INGESTION")

    
    