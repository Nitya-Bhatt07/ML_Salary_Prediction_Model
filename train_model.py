import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("Salary_data2.csv")

print(data.head())  # check columns

# Features & target
X = data[['YearsExperience']]
y = data['Salary']

# Train model
model = LinearRegression()
model.fit(X, y) 

# Save model
file = open("Salary_model.pkl", "wb")
pickle.dump(model, file)

print("Model created successfully and saved as Salary_model.pkl")