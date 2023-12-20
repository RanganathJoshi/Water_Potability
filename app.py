from src.WaterPotabilityClassification.Pipeline.Prediction import PredictPipleline,customData
from flask import Flask,request,render_template,jsonify

app=Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data=customData(
            ph=float(request.form.get('ph')),
            Hardness=float(request.form.get('Hardness')),
            Solids=float(request.form.get('Solids')),
            Chloramines=float(request.form.get('Chloramines')),
            Sulfate=float(request.form.get('Sulfate')),
            Conductivity=float(request.form.get('Conductivity')),
            Organic_carbon=float(request.form.get('Organic_carbon')),
            Trihalomethanes=float(request.form.get('Trihalomethanes')),
            Turbidity=float(request.form.get('Turbidity')),
            Potability=float(request.form.get('Potability')))

        final_data=data.get_data_as_dataframe()

        predict_pipeline=PredictPipleline()
        prediction=predict_pipeline.predict(final_data)
        if prediction==0.0:
            result="Safe Drinking Water"
        else:
            result="Not safe to drink"
        
        return render_template('result.html',final_result=result)
    

if __name__=="__main__":
        app.run()
        









        