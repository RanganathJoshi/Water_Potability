import os
import sys
from pathlib import Path
import pandas as pd
from src.WaterPotabilityClassification.logger import logging
from src.WaterPotabilityClassification.Exception import customexception
from src.WaterPotabilityClassification.utils.utils import load_object

class PredictPipleline:
    def __init__(self):
        pass
    def predict(self,feature):
        try:
            preprocessor_path=Path('artifacts/data_transformation/processor.pkl')
            model_path=Path('artifacts/model_trainer/model.pkl')


            processor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled_data=processor.transform(feature)
            prediction=model.predict(scaled_data)

            return prediction
        
        except Exception as e:
            raise customexception(e,sys)
        

class customData:
    def __init__(self,ph:float, Hardness:float, Solids:float, Chloramines:float, Sulfate:float, Conductivity:float,
                Organic_carbon:float, Trihalomethanes:float, Turbidity:float,Potability:bool):
        self.ph=ph 
        self.Hardness=Hardness
        self.Solids=Solids
        self.Chloramines=Chloramines
        self.Sulfate=Sulfate
        self.Conductivity=Conductivity
        self.Organic_carbon=Organic_carbon
        self.Trihalomethanes=Trihalomethanes
        self.Turbidity=Turbidity
        self.Potability=Potability

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                    'ph':[self.ph],
                    'Hardness':[self.Hardness],
                    'Solids':[self.Solids],
                    'Chloramines':[self.Chloramines],
                    'Sulfate':[self.Sulfate],
                    'Conductivity':[self.Conductivity],
                    'Organic_carbon':[self.Organic_carbon],
                    'Trihalomethanes':[self.Trihalomethanes],
                    'Turbidity':[self.Turbidity],
                    'Potability':[self.Potability]
                } 
            df=pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created")
            return df
        except Exception as e:
            raise customexception(e,sys)           