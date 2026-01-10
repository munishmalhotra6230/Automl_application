from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="AutoML 2026 Production API")

# Global model load (Startup par ek hi baar)
MODEL_BUNDLE = joblib.load("model_registry/latest/AutoML_Model.pkl")

# Dynamic Input Schema (Pydantic)
class ModelInput(BaseModel):
    # User ko yahan key-value pair mein data bhejna hoga
    data: dict 

@app.get("/")
def home():
    return {"message": "AutoML API is Live", "version": "2026.v1"}

@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        # 1. Convert input to DataFrame
        df = pd.DataFrame([input_data.data])
        
        # 2. Prediction
        prediction = MODEL_BUNDLE.predict(df)
        
        return {
            "status": "success",
            "prediction": prediction.tolist()[0],
            "model_version": "v1.0"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Run command: uvicorn main:app --reload
