import joblib
import pandas as pd
from app.schemas import LoanApplication, PredictionResponse

#load pretrained model
model = joblib.load("models/lightgbm.joblib")

def predict(application: LoanApplication):
    '''
    Take loan application as argument, convert to a dataframe, 
    calculate binary prediction (default/not default),
    probabilty of default, and calculate risk level based on that probability.
    '''
    
    df = pd.DataFrame([application.model_dump()])

    probability = model.predict_proba(df)[0, 1]

    if probability > 0.6:
        risk_level = 'High'
    elif probability > 0.3:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    return PredictionResponse(
    prediction=int(model.predict(df)[0]),
    probability=float(probability),
    risk_level=risk_level)