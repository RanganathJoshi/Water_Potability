import pandas as pd
import numpy as np
from src.WaterPotabilityClassification.logger import logging
from src.WaterPotabilityClassification.Exception import customexception
from src.WaterPotabilityClassification.constants import *
import os
import sys
import yaml
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from src.WaterPotabilityClassification.utils.utils import read_yaml,download_data,unzip_data

class DataConfigFile:
    config_path:Path=CONFIG_FILE_PATH
    contents=read_yaml(config_path)
    data_ingestion=contents.get('data_ingestion')
    os.makedirs(data_ingestion.get('unzip'),exist_ok=True)

class DataIngestioWorkflow:
    def __init__(self):
        self.ingestion_config=DataConfigFile()

    def initiate_data_ingestion(self):
        logging.info("DataIngestion workflow started")
        try:
            source_file:Path=self.ingestion_config.data_ingestion.get('source')
            destination_path:Path=self.ingestion_config.data_ingestion.get('raw_data')
            unzip_file:Path=self.ingestion_config.data_ingestion.get('unzip')
            zip_path=download_data(source_file,destination_path)
            unzip_data(zip_path,unzip_file)
            logging.info("Data downloaded from source and extracted it")
            data_path=Path("artifacts/data_ingestion/water_potability.csv")
            data=pd.read_csv(data_path)

            logging.info("Splitting the data into train and test data and storing it in artifacts folder")

            train_data,valid_data=train_test_split(data,test_size=0.15)
            logging.info("train test split completed")
            train_data.to_csv(self.ingestion_config.data_ingestion.get('train_data'))
            valid_data.to_csv(self.ingestion_config.data_ingestion.get('valid_data'))

            train_path:Path=self.ingestion_config.data_ingestion.get('train_data')
            valid_data:Path=self.ingestion_config.data_ingestion.get('valid_data')

            return (train_path,valid_data)
        
        except Exception as e:
            logging.info("error occured with initiating data ingestion")
            raise customexception(e,sys)

