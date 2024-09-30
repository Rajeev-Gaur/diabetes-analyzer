from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from config import RANDOM_STATE

print("train_model.py is being imported")  # Debugging statement

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Prepare the classification report
    report = classification_report(y_test, predictions)
    
    print("Accuracy:", accuracy)
    print(report)
    
    print("Attempting to save evaluation results...")
    # Save evaluation results to a text file
    with open('output/evaluation_results.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(report)
    print("Evaluation results saved.")


def save_model(model, filepath):
    joblib.dump(model, filepath)


