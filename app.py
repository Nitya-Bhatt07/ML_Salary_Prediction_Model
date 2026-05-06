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

from fastapi.responses import FileResponse

@app.get("/ui")
def serve_ui():
    return FileResponse("index.html")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)