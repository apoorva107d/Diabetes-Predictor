Diabetes Prediction Using Machine Learning

Introduction

This project implements a Diabetes Prediction System using Support Vector Machine (SVM), a machine learning algorithm. The model analyzes medical data to determine whether an individual is diabetic based on input health parameters. The dataset used is the PIMA Indian Diabetes Dataset, a well-known dataset for diabetes prediction.

Features

Data Preprocessing: Standardization of features to improve model accuracy.

Machine Learning Model: Utilizes SVM (Support Vector Machine) with a linear kernel.

Train-Test Split: Splits data into 80% training and 20% testing sets.

Model Evaluation: Calculates accuracy scores for training and test data.

Prediction System: Accepts user input to predict if a person is diabetic.

Dataset

The dataset contains medical attributes such as:

Pregnancies: Number of times pregnant

Glucose Level: Plasma glucose concentration

Blood Pressure: Diastolic blood pressure (mm Hg)

Skin Thickness: Triceps skin fold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index (weight in kg/(height in m)^2)

Diabetes Pedigree Function: A function that represents genetic diabetes risk

Age: Age of the individual in years

Installation and Usage

Prerequisites

Ensure you have Python installed along with the required libraries:

pip install numpy pandas scikit-learn

Running the Project

Load the dataset: diabetes.csv

Train the model using the script.

Input new patient data to predict diabetes.

Example Prediction

input_data = (5,166,72,19,175,25.8,0.587,51)
prediction = classifier.predict(std_data)
print('Diabetic' if prediction[0] == 1 else 'Not Diabetic')

Model Performance

Training Accuracy: ~80%

Testing Accuracy: ~76%

Future Improvements

Implement additional ML models (Random Forest, Neural Networks)

Deploy as a web application

Improve feature selection for better accuracy

