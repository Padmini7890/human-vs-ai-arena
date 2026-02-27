import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

# ===============================
# LOAD FULL DATASET
# ===============================

df = pd.read_csv("personality_dataset.csv")

# Keep selected features
df = df[[
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Personality"
]]

# Convert categorical to numeric
df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})
df["Personality"] = df["Personality"].map({"Introvert": 1, "Extrovert": 0})

# ===============================
# INSIGHT GENERATION (FULL DATA)
# ===============================

# Median grouping
alone_median = df["Time_spent_Alone"].median()
outside_median = df["Going_outside"].median()
friends_median = df["Friends_circle_size"].median()

df["Alone_Group"] = df["Time_spent_Alone"].apply(lambda x: "High" if x >= alone_median else "Low")
df["Outside_Group"] = df["Going_outside"].apply(lambda x: "High" if x >= outside_median else "Low")
df["Friends_Group"] = df["Friends_circle_size"].apply(lambda x: "High" if x >= friends_median else "Low")

# Insight 1: Stage Fear + Drained
group1 = df.groupby(["Stage_fear", "Drained_after_socializing"])["Personality"].mean() * 100
max_intro_1 = group1.max()
min_intro_1 = group1.min()

# Insight 2: Time Alone + Going Outside
group2 = df.groupby(["Alone_Group", "Outside_Group"])["Personality"].mean() * 100
max_intro_2 = group2.max()
min_intro_2 = group2.min()

# Insight 3: Friends Circle + Social Events
group3 = df.groupby(["Friends_Group", "Social_event_attendance"])["Personality"].mean() * 100
max_intro_3 = group3.max()
min_intro_3 = group3.min()

insights = {
    "description": {
        "Time_spent_Alone": "How many hours a person prefers to spend alone.",
        "Stage_fear": "Fear of speaking or performing in front of people (Yes/No).",
        "Social_event_attendance": "How often a person attends social events.",
        "Going_outside": "How frequently a person goes outside.",
        "Drained_after_socializing": "Feels tired after social interaction (Yes/No).",
        "Friends_circle_size": "Number of close friends."
    },
    "insight_1": f"People who experience stage fear and feel drained after socializing showed introvert tendencies as high as {max_intro_1:.1f}%, while the opposite pattern dropped to just {min_intro_1:.1f}%.",
    "insight_2": f"When someone spends a lot of time alone and rarely goes outside, introversion reached {max_intro_2:.1f}%, compared to only {min_intro_2:.1f}% in more outgoing lifestyles.",
    "insight_3": f"A smaller friend circle combined with lower social event attendance pushed introversion up to {max_intro_3:.1f}%, while highly social individuals showed only {min_intro_3:.1f}% introvert patterns."
}

with open("personality_insights.json", "w") as f:
    json.dump(insights, f, indent=4)

print("Personality insights generated and saved.")

# ===============================
# MODEL TRAINING
# ===============================

X = df[[
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size"
]]

y = df["Personality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Personality Model Accuracy:", accuracy)

joblib.dump(model, "personality_model.pkl")

test_df = X_test.copy()
test_df["Personality"] = y_test
test_df.to_csv("personality_test.csv", index=False)

print("Training complete. Model, test data, and insights saved.")