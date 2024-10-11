# Heart_Disease_Detection_Model
This is a model which uses machine learning to detect or predict the likelihood  of heart based on set of patient data


This repository contains the machine learning model developed to predict the likelihood of heart disease based on patient health data. The model leverages a Random Forest Classifier algorithm to provide accurate predictions based on key medical indicators.

# Overview

Heart disease is a leading cause of death worldwide, and early detection can significantly improve outcomes. This model is trained using a dataset of medical records and is designed to assist in early diagnosis by predicting whether a patient is at risk of heart disease.

# Key Features
Random Forest Classifier: A robust algorithm that combines multiple decision trees to improve prediction accuracy.
Input Parameters: Age, cholesterol level, blood pressure, and other key health indicators.
Interactive API: The model is designed to be consumed by a web application, allowing for easy integration with front-end interfaces like Blazor WebAssembly.
Metrics Reporting: Provides precision, accuracy, F1 score, and other metrics to evaluate the model's performance.


# Model Details
Algorithm: Random Forest Classifier
Input: A JSON object containing the following parameters:Age ,Sex ,ChestPainType, RestingBP , Cholesterol ,FastingBS ,RestingECG , MaxHR , ExerciseAngina ,OldPeak ,ST_Slope
Output: A prediction indicating the probability of heart disease (0 for no risk, 1 for high risk).

# Installation and Setup

Clone this repository:
git clone https://github.com/yourusername/heart-disease-detection-model.git

