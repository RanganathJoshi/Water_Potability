import sys
import pandas as pd
import numpy as np
from src.Water_Potability_Classification.constants import *
from src.Water_Potability_Classification.logger import logging
from src.Water_Potability_Classification.Exception import customexception
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
from dataclasses import dataclass
from src.Water_Potability_Classification.utils.utils import read_yaml,save_object

@dataclass
class DataTransformationConfig:
    config_path=CONFIG_FILE_PATH
    contents=read_yaml(config_path)
    data_transformation=contents.get('data_transformation')
    os.makedirs(data_transformation.get('root_path'),exist_ok=True)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def initiate_data_transformation(self):
        try:
            logging.info("Starting Data Transformation")
            columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes', 'Turbidity']

            logging.info("Pipeline Initiated")

            num_pipeline=Pipeline(steps=[
                ('Imputer',SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())])

            processor=ColumnTransformer([('numpipeline',num_pipeline,columns)],remainder='passthrough')

            return processor
        
        except Exception as e:
            logging.info("Exception occured while creating processor obj")


    def apply_transformationa(self,train_path,valid_path):
        try:
            train=pd.read_csv(train_path)
            valid=pd.read_csv(valid_path)
            logging.info("Datasets loaded for transformations")

            train_df=train.drop(columns=['Unnamed: 0'])
            valid_df=valid.drop(columns=['Unnamed: 0'])

            processing_obj=self.initiate_data_transformation()

            train_transform=pd.DataFrame(processing_obj.fit_transform(train_df), columns=train_df.columns)
            valid_transform=pd.DataFrame(processing_obj.transform(valid_df),columns=valid_df.columns)

            if not(os.path.exists(self.data_transformation_config.data_transformation.get('train_data_transform'))) or not(os.path.exists(self.data_transformation_config.data_transformation.get('vaid_data_transform'))):
                train_transform.to_csv(self.data_transformation_config.data_transformation.get('train_data_transform'))
                valid_transform.to_csv(self.data_transformation_config.data_transformation.get('vaid_data_transform'))
            
            preprocessot_obj_path=self.data_transformation_config.data_transformation.get('processor_obj_path')

            save_object(preprocessot_obj_path,processing_obj)

            return (self.data_transformation_config.data_transformation.get('train_data_transform'),self.data_transformation_config.data_transformation.get('vaid_data_transform'))
        

        except Exception as e:
            logging.info("Error occured while applying transformation")
            raise customexception(e,sys)




