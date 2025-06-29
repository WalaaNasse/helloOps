import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../data-storage/breast_dataset.csv")
label_class = LabelEncoder()
data["Class"] = label_class.fit_transform(data["Class"])

app = FastAPI()
model = joblib.load("../model-storage/forest.joblib")

@app.get("/")
async def root():
    return {"message:": "Welcome to our API page!"}

@app.post("/predict/")
async def predict_cancer(data: dict): 
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    class_name = label_class.inverse_transform(prediction)[0]
    return {"class": class_name}