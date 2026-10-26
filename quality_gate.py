
import json
import joblib
from sklearn.metrics import accuracy_score
from train_model import X_test, y_test

# Load the trained model
model = joblib.load("student_result_model.pkl")

# Check model accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model metrics
metrics = {
    "accuracy": round(float(accuracy), 2),
    "minimum_accuracy": 0.70,
    "status": "PASSED" if accuracy >= 0.70 else "FAILED"
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print(json.dumps(metrics, indent=4))

if accuracy < 0.70:
    raise SystemExit("Quality gate failed: accuracy is below 70%.")

print("Quality gate passed!")
