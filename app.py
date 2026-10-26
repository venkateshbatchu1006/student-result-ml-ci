
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "student_result_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Result Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    global model

    if model is None:
        return jsonify({"error": "Model not found"}), 503

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please provide JSON input"}), 400

    features = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    if not all(feature in data for feature in features):
        return jsonify({
            "error": "All four input fields are required"
        }), 400

    try:
        values = [float(data[feature]) for feature in features]
    except (ValueError, TypeError):
        return jsonify({
            "error": "Input values must be numbers"
        }), 400

    if not all(0 <= value <= 100 for value in values):
        return jsonify({
            "error": "All input values must be between 0 and 100"
        }), 400

    input_data = pd.DataFrame([values], columns=features)
    prediction = int(model.predict(input_data)[0])

    result = "PASS" if prediction == 1 else "FAIL"

    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
