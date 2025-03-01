import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading the diabetes dataset
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the SVM Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluating the model
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Training Accuracy:', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Test Accuracy:', test_data_accuracy)

# Making a predictive system
input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardizing input data
std_data = scaler.transform(input_data_reshaped)

# Making a prediction
prediction = classifier.predict(std_data)
print('Diabetes Prediction:', 'Diabetic' if prediction[0] == 1 else 'Not Diabetic')
