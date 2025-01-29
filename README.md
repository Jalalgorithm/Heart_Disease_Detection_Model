# AI-Enhanced Heart Disease Detection Model

## Overview
This project is an AI-powered heart disease detection system that leverages machine learning to provide accurate medical assessments. It uses a **Random Forest Classifier** trained on a dataset containing heart disease risk factors. The model is developed using **Jupyter Notebook** and is designed to predict the likelihood of heart disease based on user input parameters.

## Features
- **Machine Learning Algorithm:** Random Forest Classifier
- **Training Environment:** Jupyter Notebook
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
- **Model Persistence:** Saved as a `.pkl` file for reuse
- **API Integration:** Can be deployed as a Flask API for real-world applications

## Dataset
The model is trained on a dataset containing relevant features such as:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- ST Slope

## Installation
To set up and run the project locally, follow these steps:

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/Jalalgorithm/Heart_Disease_Detection_Model.git
   cd heart_disease_detection_model
   ```
2. **Install Dependencies:**
   Make sure you have Python installed, then install required libraries:
   ```sh
   pip install -r requirements.txt
   ```
3. **Run Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```
   Open the `.ipynb` file and execute the cells step by step.

## Usage
1. Open `heart_disease_detection.ipynb` in Jupyter Notebook.
2. Load the dataset and preprocess the data.
3. Train the Random Forest model.
4. Evaluate model performance using accuracy, precision, recall, and F1-score.
5. Save the trained model as a `.pkl` file for future predictions.
6. Use the model to make predictions by inputting patient data.

## Model Performance
After training and evaluation, the model achieved the following scores:
- **Accuracy:** 90%
- **Precision:** 89%
- **Recall:** 89%
- **F1-Score:** 89%


## Deployment
To deploy the model as an API:
1. Use Flask to create a RESTful API.
2. Load the saved model (`.pkl` file) in the API.
3. Accept user input via JSON.
4. Return predictions in JSON format.

## Contribution
If you'd like to contribute:
- Fork the repository
- Create a new branch (`feature-branch`)
- Commit changes
- Open a pull request

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contact
For any inquiries, feel free to reach out:
- **Email:** ahmeedabduljalal@gmail.com
- **GitHub:** [jalalgorithm](https://github.com/jalalgorithm)

