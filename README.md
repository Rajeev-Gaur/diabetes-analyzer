# Diabetes Analyzer

## Overview

The Diabetes Analyzer is a machine learning-based application that predicts whether an individual has diabetes based on health metrics. It uses a trained Random Forest model to analyze user inputs and provide clear predictions.

## Features

- User-friendly interface for inputting health metrics.
- Predicts diabetes risk based on various health indicators.
- Clear output messages indicating whether the patient is diabetic or non-diabetic.
- Saves evaluation results and the trained model for future use.

## Requirements

- Python 3.6+
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-analyzer.git
   cd diabetes-analyzer
2. Create a virtual environment:
    python -m venv .venv
 
3. Activate the virtual environment:
    .venv\Scripts\activate

4. Install the required packages:
    pip install -r requirements.txt


Usage
1. Run the main application:  python main.py
2. Follow the prompts to enter the required health metrics:

    Number of pregnancies
    Glucose level
    Blood pressure
    Skin thickness
    Insulin level
    BMI
    Diabetes pedigree function
    Age
3. Receive a prediction indicating whether the patient is diabetic or non-diabetic.

Data
The model is trained using the Pima Indians Diabetes Database available on Kaggle.

Evaluation
The evaluation results of the model's performance can be found in the output/evaluation_results.txt file after running the training process.

License
This project is licensed under the MIT License. See the LICENSE.txt file for details.

Acknowledgments
The dataset used for training and evaluation is sourced from Kaggle.
This project utilizes the Scikit-learn library for machine learning.