import joblib
import pandas as pd
from config import MODEL_OUTPUT_PATH

# Load the model
model = joblib.load(MODEL_OUTPUT_PATH)
print("Model loaded successfully!")

# Example input based on the first row of your dataset
X_new = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]],
                     columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

# Make predictions
predictions = model.predict(X_new)
print("Predictions:", predictions)

