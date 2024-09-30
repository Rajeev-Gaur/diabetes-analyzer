import pandas as pd
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE, DATA_PATH

print("preprocessing_data.py is being imported")  # Debugging statement

def load_and_preprocess_data(filepath=DATA_PATH):
    data = pd.read_csv(filepath)
    if data.isnull().sum().any():
        raise ValueError("Data contains missing values!")
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

