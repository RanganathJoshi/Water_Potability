from pathlib import Path
import os

packagename="Water_Potability_Classification"

list_files=[
    "github/workflow/.gitkeep",
    f"src/{packagename}/Components/__init__.py",
    f"src/{packagename}/Components/DataIngestion.py",
    f"src/{packagename}/Components/FeatureEngineering.py",
    f"src/{packagename}/Components/ModelTraining.py",
    f"src/{packagename}/Pipeline/__init__.py",
    f"src/{packagename}/Pipeline/Training.py",
    f"src/{packagename}/Pipeline/Prediction.py",
    f"src/{packagename}/logger.py",
    f"src/{packagename}/Exception.py",
    f"src/{packagename}/utils/__init__.py",
    "requirements.txt",
    "Setup.py",
    "init_setup.sh",
    "notebooks/research.ipynb",
    "notebooks/data/.gitkeep"

]

for file_name in list_files:
    file=Path(file_name)
    dir,files=os.path.split(file)

    if dir!='':
        os.makedirs(dir,exist_ok=True)
    
    if (not os.path.exists(file_name)) or (os.path.getsize(file_name)==0):
        with open(file_name,'w') as f:
            pass
    else:
        print("file already exists")
