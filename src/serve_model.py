from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Job Application Success Predictor",
    description="Predicts the likelihood of a job application being successful based on resume features.",
    version="1.0.0"
)

# Load trained model
model = joblib.load("models/best_model.pkl")

# Define input schema
class ResumeFeatures(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "✅ Job Application Success Predictor API is running!"}

@app.post("/predict")
def predict(data: ResumeFeatures):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    return {
        "prediction": int(prediction),
        "success_probability": round(float(probability), 4)
    }
