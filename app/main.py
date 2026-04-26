from fastapi import FastAPI
from app.predictor import predict
from app.schemas import LoanApplication, PredictionResponse

app = FastAPI(
    title='Loanguard',
    description = "Credit default prediction API"
)

@app.get("/health")
def health_check():
    #check that api is working
    return {'status': 'ok'}


@app.post("/predict", response_model=PredictionResponse)
def predict_default(application: LoanApplication):
    """
    Accept a loan application and return default prediction,
    default probability, and risk level.
    """
    return predict(application)

@app.get("/")
def root():
    return {"message": "LoanGuard API is running"}