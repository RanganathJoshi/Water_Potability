import os
import yaml
import mlflow
import numpy as np
import pandas as pd
import mlflow.sklearn
from mlflow.data.pandas_dataset import PandasDataset
from pathlib import Path
from urllib.parse import urlparse
from src.Water_Potability_Classification.utils.utils import load_object
from mlflow.models import infer_signature

class SaveModel:
    def __init__(self,model_path,scaled_data_path):
        self.model_path=model_path
        self.data_path=scaled_data_path

    def log_into_mlflow(self,mlflow_uri):
        data=pd.read_csv(self.data_path)
        dataset: PandasDataset = mlflow.data.from_pandas(data)
        mlflow.set_tracking_uri(mlflow_uri)
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
        model=load_object(self.model_path)
        with mlflow.start_run():
            if tracking_url_type_store!='file':
                mlflow.log_input(dataset, context="training data")
                mlflow.sklearn.log_model(model,"model",registered_model_name="Best Model ")
            else:
                mlflow.sklearn.log_model(model,"model") 


            