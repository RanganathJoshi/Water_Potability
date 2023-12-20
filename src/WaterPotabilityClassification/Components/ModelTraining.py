from src.WaterPotabilityClassification.logger import logging
from src.WaterPotabilityClassification.Exception import customexception
import pandas
import pickle
from pathlib import Path
import sys
from src.WaterPotabilityClassification.utils.utils import read_yaml,save_object,evaluate_model
import pandas as pd
import os
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score
from src.WaterPotabilityClassification.constants import *
from sklearn.linear_model import LogisticRegression,RidgeClassifier,Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

@dataclass 
class Mode_training_config:
    config_path=CONFIG_FILE_PATH
    param_path=PARAM_FILE_PATH
    contents=read_yaml(config_path)
    param_content=read_yaml(param_path)
    params=param_content.get('parameters')
    model_trainer=contents.get('model_trainer')
    os.makedirs(model_trainer.get('root_path'),exist_ok=True)


class Model_Trainer:
    def __init__(self):
        self.config=Mode_training_config()

    def initiate_model_training(self,train_path,valid_path):
        try:
            logging.info("Model training Initiated")
            train_data=pd.read_csv(train_path)
            valid_data=pd.read_csv(valid_path)
            print(train_data.shape)
            print(valid_data.shape)
            x_train,x_test,y_train,y_test=(train_data.iloc[:,:-1],
                                           valid_data.iloc[:,:-1],
                                           train_data.iloc[:,-1],
                                           valid_data.iloc[:,-1])
            
            models={'knn':KNeighborsClassifier(),
                'Adaboost':AdaBoostClassifier(),
                'GradientBoost':GradientBoostingClassifier(),
                'SVC':SVC(),
                'Decision Tree':DecisionTreeClassifier(),
                'RandomForest': RandomForestClassifier(),
                'BaggingClassifierknn':BaggingClassifier(estimator=KNeighborsClassifier()),
                'BaggingClassifierSVC':BaggingClassifier(estimator=SVC())
    
            }

            test_model_report,train_model_report=evaluate_model(models,x_train,x_test,y_train,y_test,self.config.params)
            print(train_model_report)
            print("\n======================================================")
            logging.info("Models trained and reports generated")
            best_model_score=max(list((test_model_report.values())))
            best_model_name = list(test_model_report.keys())[
                list(test_model_report.values()).index(best_model_score)
            ]
        
            best_model=models[best_model_name]


            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f"Model Summary :{test_model_report}")
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(self.config.model_trainer.get('model_path'),
                        obj=best_model)
            
            return self.config.model_trainer
        
        except Exception as e:
            logging.info("Error while occuring model training")
            raise customexception(e,sys)
    





