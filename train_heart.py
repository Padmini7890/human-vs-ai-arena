import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

# ===============================
# LOAD FULL DATASET
# ===============================

df = pd.read_csv("heart_dataset.csv")

# Keep selected features
df = df[[
    "age",
    "cp",
    "thalach",
    "exang",
    "oldpeak",
    "ca",
    "target"
]]

# ===============================
# INSIGHT GENERATION (FULL DATA)
# ===============================

# Median-based grouping for numeric variables
age_median = df["age"].median()
thalach_median = df["thalach"].median()
oldpeak_median = df["oldpeak"].median()

df["Age_Group"] = df["age"].apply(lambda x: "High" if x >= age_median else "Low")
df["HR_Group"] = df["thalach"].apply(lambda x: "High" if x >= thalach_median else "Low")
df["Oldpeak_Group"] = df["oldpeak"].apply(lambda x: "High" if x >= oldpeak_median else "Low")

# Insight 1: Chest Pain + Exercise Angina
group1 = df.groupby(["cp", "exang"])["target"].mean() * 100

# Select most risky combination dynamically
max_risk_1 = group1.max()
min_risk_1 = group1.min()

# Insight 2: ST Depression + Major Vessels
group2 = df.groupby(["Oldpeak_Group", "ca"])["target"].mean() * 100

max_risk_2 = group2.max()
min_risk_2 = group2.min()

# Insight 3: Heart Rate + ST Depression
group3 = df.groupby(["HR_Group", "Oldpeak_Group"])["target"].mean() * 100

max_risk_3 = group3.max()
min_risk_3 = group3.min()

# Create readable insights
insights = {
    "description": {
        "age": "Age of the patient.",
        "cp": "Type of chest pain experienced.",
        "thalach": "Maximum heart rate achieved.",
        "exang": "Chest pain during exercise (1 = Yes, 0 = No).",
        "oldpeak": "ST depression indicating heart stress level.",
        "ca": "Number of major vessels blocked."
    },
    "insight_1": f"Certain combinations of chest pain type and exercise-induced angina showed disease presence as high as {max_risk_1:.1f}%, while some combinations showed as low as {min_risk_1:.1f}%.",
    "insight_2": f"Patients with higher ST depression and more blocked vessels had disease rates up to {max_risk_2:.1f}%, compared to as low as {min_risk_2:.1f}% in lower-risk groups.",
    "insight_3": f"When heart rate and ST depression levels were both high-risk, disease occurrence reached {max_risk_3:.1f}%, whereas low-risk combinations dropped to {min_risk_3:.1f}%."
}

# Save insights
with open("heart_insights.json", "w") as f:
    json.dump(insights, f, indent=4)

print("Heart insights generated and saved.")

# ===============================
# MODEL TRAINING (80/20 SPLIT)
# ===============================

X = df[[
    "age",
    "cp",
    "thalach",
    "exang",
    "oldpeak",
    "ca"
]]

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Heart Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "heart_model.pkl")

# Save test data
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("heart_test.csv", index=False)

print("Training complete. Model, test data, and insights saved.")