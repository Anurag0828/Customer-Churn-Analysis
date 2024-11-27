import pickle
# Load the model
filename = 'model.sav'
load_model = pickle.load(open(filename, 'rb'))
from pydantic import BaseModel

class ModelInput(BaseModel):
    SeniorCitizen: int
    MonthlyCharges: float
    TotalCharges: float
    gender_Female: int
    gender_Male: int
    Partner_No: int
    Partner_Yes: int
    Dependents_No: int
    Dependents_Yes: int
    PhoneService_No: int
    PhoneService_Yes: int
    MultipleLines_No: int
    MultipleLines_No_phone_service: int
    MultipleLines_Yes: int
    InternetService_DSL: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    OnlineSecurity_No: int
    OnlineSecurity_No_internet_service: int
    OnlineSecurity_Yes: int
    OnlineBackup_No: int
    OnlineBackup_No_internet_service: int
    OnlineBackup_Yes: int
    DeviceProtection_No: int
    DeviceProtection_No_internet_service: int
    DeviceProtection_Yes: int
    TechSupport_No: int
    TechSupport_No_internet_service: int
    TechSupport_Yes: int
    StreamingTV_No: int
    StreamingTV_No_internet_service: int
    StreamingTV_Yes: int
    StreamingMovies_No: int
    StreamingMovies_No_internet_service: int
    StreamingMovies_Yes: int
    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int
    PaperlessBilling_No: int
    PaperlessBilling_Yes: int
    PaymentMethod_Bank_transfer_automatic: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int
    tenure_group_1_12: int
    tenure_group_13_24: int
    tenure_group_25_36: int
    tenure_group_37_48: int
    tenure_group_49_60: int
    tenure_group_61_72: int


from fastapi import FastAPI
import numpy as np
import pandas as pd

app = FastAPI()

@app.post("/predict")
def predict(input_data: ModelInput):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Predict using the loaded model
    prediction = load_model.predict(input_df)
    
    return {"prediction": int(prediction[0])}
