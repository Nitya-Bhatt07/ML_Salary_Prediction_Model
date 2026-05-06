from fastapi import FastAPI
import pickle

app = FastAPI()

# Load model
file = open("Salary_model.pkl", "rb")
model = pickle.load(file)

@app.get("/")
def home():
    return {"message": "Salary Prediction API is running 🚀"}

@app.get("/predict")
def predict(experience: float):
    prediction = model.predict([[experience]])
    return {"salary": float(prediction[0])}