from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# MLflow olmadan direkt path üzerinden model yükleniyor
model = mlflow.pyfunc.load_model("model")

class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: float
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(input: ChurnInput):
    input_df = pd.DataFrame([input.dict()])
    prediction = model.predict(input_df)
    return {"churn_probability": float(prediction[0])}
