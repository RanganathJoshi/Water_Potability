from src.WaterPotabilityClassification.Components.DataIngestion import DataIngestioWorkflow
from src.WaterPotabilityClassification.Components.FeatureEngineering import DataTransformation
from src.WaterPotabilityClassification.Components.ModelTraining import Model_Trainer
import os
import pandas as pd
import numpy as np
from src.WaterPotabilityClassification.logger import logging
from src.WaterPotabilityClassification.Exception import customexception
obj=DataIngestioWorkflow()
train_path,valid_path=obj.initiate_data_ingestion()
transform=DataTransformation()
train_data_path,valid_data_path=transform.apply_transformationa(train_path,valid_path)
print(train_data_path,valid_data_path)
model_training=Model_Trainer()
model_training.initiate_model_training(train_data_path,valid_data_path)
mlflow_uri="https://dagshub.com/RanganathJoshi/Water_Potability.mlflow"
