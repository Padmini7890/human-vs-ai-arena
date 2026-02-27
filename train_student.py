import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

# ===============================
# Load Full Dataset
# ===============================

df = pd.read_csv("student_dataset.csv")

# Keep only required columns
df = df[[
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "Internet_Access_at_Home",
    "Extracurricular_Activities",
    "Pass_Fail"
]]

# Convert Yes/No to 1/0
df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map({"Yes": 1, "No": 0})
df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"Yes": 1, "No": 0})
df["Pass_Fail"] = df["Pass_Fail"].map({"Pass": 1, "Fail": 0})

# ===============================
# 🔎 INSIGHT GENERATION (Full Dataset)
# ===============================

# Compute medians for grouping
study_median = df["Study_Hours_per_Week"].median()
attendance_median = df["Attendance_Rate"].median()
past_score_median = df["Past_Exam_Scores"].median()

# Create High/Low categories
df["Study_Group"] = df["Study_Hours_per_Week"].apply(lambda x: "High" if x >= study_median else "Low")
df["Attendance_Group"] = df["Attendance_Rate"].apply(lambda x: "High" if x >= attendance_median else "Low")
df["Past_Group"] = df["Past_Exam_Scores"].apply(lambda x: "High" if x >= past_score_median else "Low")

# Insight 1: Study Hours + Internet Access
group1 = df.groupby(["Study_Group", "Internet_Access_at_Home"])["Pass_Fail"].mean() * 100

high_study_internet = group1.get(("High", 1), 0)
low_study_no_internet = group1.get(("Low", 0), 0)

# Insight 2: Attendance + Past Scores
group2 = df.groupby(["Attendance_Group", "Past_Group"])["Pass_Fail"].mean() * 100

high_att_high_past = group2.get(("High", "High"), 0)
low_att_low_past = group2.get(("Low", "Low"), 0)

# Insight 3: Study Hours + Attendance
group3 = df.groupby(["Study_Group", "Attendance_Group"])["Pass_Fail"].mean() * 100

high_study_high_att = group3.get(("High", "High"), 0)
low_study_low_att = group3.get(("Low", "Low"), 0)

# Create readable insight statements
insights = {
    "insight_1": f"Students with high study hours and internet access passed {high_study_internet:.1f}% of the time, while students with low study hours and no internet passed {low_study_no_internet:.1f}%.",
    
    "insight_2": f"Students with high attendance and high past exam scores passed {high_att_high_past:.1f}% of the time, whereas students with low attendance and low past scores passed {low_att_low_past:.1f}%.",
    
    "insight_3": f"When both study hours and attendance were high, pass rate reached {high_study_high_att:.1f}%, but when both were low, pass rate dropped to {low_study_low_att:.1f}%."
}

# Save insights
with open("student_insights.json", "w") as f:
    json.dump(insights, f, indent=4)

print("Insights generated and saved.")

# ===============================
# MODEL TRAINING (80/20 Split)
# ===============================

X = df[[
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "Internet_Access_at_Home",
    "Extracurricular_Activities"
]]

y = df["Pass_Fail"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Student Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "student_model.pkl")

# Save test data
test_df = X_test.copy()
test_df["Pass_Fail"] = y_test
test_df.to_csv("student_test.csv", index=False)

print("Training complete. Model, test data, and insights saved.")