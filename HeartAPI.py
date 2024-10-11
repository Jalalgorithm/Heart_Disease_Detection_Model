from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Paths
csv_file_path = os.path.join("C:\\Users\\AHMEED\\Desktop\\Diagnostics\\HeartDetector", "heart.csv")
model_path = os.path.join("C:\\Users\\AHMEED\\Desktop\\Diagnostics\\HeartDetector", "heart_disease_model.pkl")
accuracy_file_path = os.path.join("C:\\Users\\AHMEED\\Desktop\\Diagnostics\\HeartDetector", "model_accuracy.txt")

# Define the categories for one-hot encoding
categories = {
    "ChestPainType": ['ATA', 'NAP', 'ASY', 'TA'],
    "Sex": ['M', 'F'],
    "RestingECG": ['Normal', 'ST', 'LVH'],
    "ExerciseAngina": ['N', 'Y'],
    "ST_Slope": ['Up', 'Flat', 'Down']
}

def train_and_save_model():
    data = pd.read_csv(csv_file_path)
    
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    X = pd.get_dummies(X, columns=categories.keys())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)

    # Save the model
    joblib.dump(rf_classifier, model_path)
    
    # Evaluate the model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save accuracy to a file
    with open(accuracy_file_path, 'w') as f:
        f.write(f"{accuracy * 100:.2f}")

    return rf_classifier, X_train.columns

def load_model():
    if not os.path.exists(model_path):
        model, features = train_and_save_model()
    else:
        model = joblib.load(model_path)
        data = pd.read_csv(csv_file_path)
        X = data.drop("HeartDisease", axis=1)
        X = pd.get_dummies(X, columns=categories.keys())
        features = X.columns
    
    return model, features

model, model_features = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input JSON data
        input_data = request.get_json()

        # Convert the JSON data to a DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, columns=categories.keys())

        # Ensure all necessary columns are present
        missing_features = set(model_features) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0
        input_df = input_df[model_features]

        # Make a prediction
        prediction = model.predict(input_df)

        # Return the prediction result
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

# @app.route('/accuracy', methods=['GET'])
# def get_accuracy():
#     try:
#         if os.path.exists(accuracy_file_path):
#             with open(accuracy_file_path, 'r') as f:
#                 accuracy = f.read()
#             return jsonify({"accuracy": accuracy})
#         else:
#             return jsonify({"error": "Accuracy file not found."}), 404
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500



# model_file_path_diab = os.path.join("C:\\Users\\AHMEED\\Desktop\\Diagnostics\\HeartDetector", "diabetes_model_xg.pkl")

# # Load your model
# model = joblib.load(model_file_path_diab)

# csv_file_path = os.path.join("C:\\Users\\AHMEED\\Desktop\\Diagnostics\\HeartDetector", "diabetes.csv")
# data = pd.read_csv(csv_file_path)
# X = data.drop("Outcome", axis=1)
# y = data["Outcome"]
# _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# @app.route('/diabetes', methods=['POST'])
# def predict_diabetes():
#     try:
#         data = request.get_json()
#         patient_df = pd.DataFrame(data, index=[0])
#         prediction = model.predict(patient_df)
#         # Replace X_test and y_test with your actual test data for accuracy calculation
#         accuracy= model.score(X_test , y_test)
#         response = {
#             'prediction': int(prediction[0]),
#             'accuracy':accuracy
#         }
#         return jsonify(response)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# model_file_path = os.path.join("C:\\Users\\AHMEED\\Desktop\\Diagnostics\\HeartDetector", "diabetes_svm_model.pkl")
# model = joblib.load(model_file_path)

# csv_file_path = os.path.join("C:\\Users\\AHMEED\\Desktop\\Diagnostics\\HeartDetector", "diabetes.csv")
# data = pd.read_csv(csv_file_path)
# X = data.drop("Outcome", axis=1)
# y = data["Outcome"]
# _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# @app.route('/diabetes-svm', methods=['POST'])
# def predict_smvdiab():
#     data = request.get_json(force=True)
#     patient_df = pd.DataFrame(data, index=[0])
#     prediction = model.predict(patient_df)
#     accuracy = model.score(X_test, y_test)  # Calculate the model's accuracy
    
#     response = {
#         'prediction': int(prediction[0]),
#         'accuracy': accuracy
#     }
#     return jsonify(response)


# model_file_path = os.path.join("C:\\Users\\AHMEED\\Desktop\\Diagnostics\\HeartDetector", "diabetes_knn_model.pkl")
# model = joblib.load(model_file_path)

# # Example X_test and y_test (you need to ensure these are accessible and properly loaded)
# csv_file_path = os.path.join("C:\\Users\\AHMEED\\Desktop\\Diagnostics\\HeartDetector", "diabetes.csv")
# data = pd.read_csv(csv_file_path)
# X = data.drop("Outcome", axis=1)
# y = data["Outcome"]
# _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# @app.route('/predict-knn', methods=['POST'])
# def predict_knn():
#     data = request.get_json(force=True)
#     patient_df = pd.DataFrame(data, index=[0])
#     prediction = model.predict(patient_df)
#     accuracy = model.score(X_test, y_test)  # Calculate the model's accuracy
    
#     response = {
#         'prediction': int(prediction[0]),
#         'accuracy': accuracy
#     }
#     return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)