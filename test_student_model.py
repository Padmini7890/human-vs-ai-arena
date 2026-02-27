import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

# Load test dataset
df = pd.read_csv("student_test.csv")

# Pick first row from test data
sample = df.iloc[0]

# Separate features and actual label
X_sample = pd.DataFrame([sample.drop("Pass_Fail")])
actual = sample["Pass_Fail"]

# Make prediction
prediction = model.predict(X_sample)[0]

# Convert numeric prediction to readable label
if prediction == 1:
    prediction_label = "PASS ✅"
else:
    prediction_label = "FAIL ❌"

if actual == 1:
    actual_label = "PASS ✅"
else:
    actual_label = "FAIL ❌"

print("Student Profile:")
print(sample.drop("Pass_Fail"))

print("\nModel Prediction:", prediction_label)
print("Actual Result:", actual_label)