
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample student dataset
data = {
    "attendance": [90, 85, 40, 35, 75, 30, 95, 60, 80, 25,
                   88, 45, 70, 20, 92, 50, 65, 15, 78, 55],
    "internal_marks": [85, 80, 35, 30, 70, 25, 95, 55, 78, 20,
                       90, 40, 65, 15, 88, 45, 60, 10, 75, 50],
    "assignment_marks": [90, 85, 40, 35, 75, 30, 98, 60, 82, 25,
                         92, 45, 72, 18, 94, 50, 68, 12, 80, 55],
    "previous_score": [88, 82, 38, 32, 72, 28, 96, 58, 80, 22,
                       91, 42, 68, 16, 90, 48, 62, 12, 77, 53],
    "result": [1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
               1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("result", axis=1)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "student_result_model.pkl")

print(f"Model trained successfully. Accuracy: {accuracy:.2f}")
print("Model saved as student_result_model.pkl")
