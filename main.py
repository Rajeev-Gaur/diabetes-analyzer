from preprocessing_data import load_and_preprocess_data
from train_model import train_model, evaluate_model, save_model
import joblib
import pandas as pd
from config import MODEL_OUTPUT_PATH

def get_input(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt))
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                print(f"Please enter a value between {min_value} and {max_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def main():
    # Load the model
    model = joblib.load(MODEL_OUTPUT_PATH)
    print("Model loaded successfully!")

    # Collect user input with validation
    pregnancies = get_input("Enter number of pregnancies: ", 0)
    glucose = get_input("Enter glucose level: ", 0, 200)
    blood_pressure = get_input("Enter blood pressure: ", 0, 150)
    skin_thickness = get_input("Enter skin thickness: ", 0, 100)
    insulin = get_input("Enter insulin level: ", 0, 900)
    bmi = get_input("Enter BMI: ", 10, 60)
    diabetes_pedigree = get_input("Enter diabetes pedigree function: ", 0, 2.5)
    age = get_input("Enter age: ", 1, 120)

    # Prepare the input DataFrame
    X_new = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, diabetes_pedigree, age]],
                         columns=["Pregnancies", "Glucose", "BloodPressure", 
                                  "SkinThickness", "Insulin", "BMI", 
                                  "DiabetesPedigreeFunction", "Age"])

    # Make predictions
    predictions = model.predict(X_new)

    # Interpret the prediction
    if predictions[0] == 1:
        result = "The patient is diabetic."
    else:
        result = "The patient is non-diabetic."

    # Output the result
    print("Diabetic result:", result)

if __name__ == "__main__":
    main()


 
