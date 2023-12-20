import yaml
import os
from pathlib import Path  
import zipfile
import urllib.request as request 
import pickle 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def read_yaml(path_to_yaml:Path):
    with open(path_to_yaml,'r') as file:
        contents=yaml.safe_load(file)

        return contents
    
def download_data(url_path:str,destination:Path):
    if not (os.path.exists(destination)):
        file,headers=request.urlretrieve(url=url_path,filename=destination)
    return destination


def unzip_data(zip_file_path,destination):
    os.makedirs(destination, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination)

    return destination


def load_object(file_path:Path):
    with open(file_path,'rb') as file:
        return pickle.load(file)


def save_object(file_path,obj):
    dir_name=os.path.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)

    with open(file_path,"wb") as file_obj:
        pickle.dump(obj,file_obj)

def evaluate_model(models,x_train,x_test,y_train,y_test,params):
    report_test={}
    report_train={}
    for i in range(len(models)):
        model=list(models.keys())[i]
        classifier=list(models.values())[i]
        param=params[model]
        gs=GridSearchCV(classifier,param,cv=3)
        gs.fit(x_train,y_train)
        classifier.set_params(**gs.best_params_)
        classifier.fit(x_train,y_train)
        y_pred_train=classifier.predict(x_train)
        y_pred=classifier.predict(x_test)
        score_test=accuracy_score(y_test,y_pred)
        score_train=accuracy_score(y_train,y_pred_train)
        report_test[model]=score_test
        report_train[model]=score_train



    return report_test,report_train

