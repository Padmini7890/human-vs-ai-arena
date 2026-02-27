import streamlit as st
import pandas as pd
import joblib
import json
import random
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Human vs AI Arena", layout="wide")


# =====================================================
# SCOREBOARD SETUP (PERSISTENT)
# =====================================================

SCORE_FILE = "scoreboard.json"

default_scores = {
    "global": {"ai_correct": 0, "ai_total": 0, "human_correct": 0, "human_total": 0},
    "student": {"ai_correct": 0, "ai_total": 0, "human_correct": 0, "human_total": 0},
    "heart": {"ai_correct": 0, "ai_total": 0, "human_correct": 0, "human_total": 0},
    "personality": {"ai_correct": 0, "ai_total": 0, "human_correct": 0, "human_total": 0}
}

if not os.path.exists(SCORE_FILE):
    with open(SCORE_FILE, "w") as f:
        json.dump(default_scores, f)

with open(SCORE_FILE, "r") as f:
    scores = json.load(f)

def save_scores():
    with open(SCORE_FILE, "w") as f:
        json.dump(scores, f, indent=4)

def calc_accuracy(correct, total):
    return (correct / total * 100) if total > 0 else 0

# =====================================================
# SESSION STATE FOR QUESTION CONTROL
# =====================================================

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "revealed" not in st.session_state:
    st.session_state.revealed = False

for ds in ["student", "heart", "personality"]:
    if f"{ds}_used_indices" not in st.session_state:
        st.session_state[f"{ds}_used_indices"] = []

# =====================================================
# TITLE
# =====================================================

st.markdown("<h1 style='text-align: center;'>🧠 Human vs AI Arena</h1>", unsafe_allow_html=True)

dataset_choice = st.selectbox(
    "Choose Your Challenge",
    ["Student Performance", "Heart Disease", "Personality Prediction"]
)

if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = dataset_choice

if st.session_state.last_dataset != dataset_choice:
    st.session_state.current_question = None
    st.session_state.revealed = False
    st.session_state.last_dataset = dataset_choice

st.divider()

# =====================================================
# GLOBAL SIDEBAR LEADERBOARD
# =====================================================

st.sidebar.header("🌍 Overall Leaderboard")

# Calculate accuracy
ai_acc = calc_accuracy(scores["global"]["ai_correct"], scores["global"]["ai_total"])
human_acc = calc_accuracy(scores["global"]["human_correct"], scores["global"]["human_total"])

st.sidebar.markdown("### 🤖 AI Performance")
st.sidebar.progress(ai_acc / 100)
st.sidebar.write(f"{scores['global']['ai_correct']} / {scores['global']['ai_total']}  ({ai_acc:.2f}%)")

st.sidebar.divider()

st.sidebar.markdown("### 👥 Human Performance")
st.sidebar.progress(human_acc / 100)
st.sidebar.write(f"{scores['global']['human_correct']} / {scores['global']['human_total']}  ({human_acc:.2f}%)")

st.sidebar.divider()

if st.sidebar.button("🔄 Reset Entire Game"):
    with open(SCORE_FILE, "w") as f:
        json.dump(default_scores, f, indent=4)
    st.session_state.current_question = None
    st.session_state.revealed = False
    st.rerun()

# =====================================================
# DATASET HANDLER
# =====================================================

def run_dataset(ds_key, model_file, test_file, target_col, vote_options, label_map):

    model = joblib.load(model_file)
    test_df = pd.read_csv(test_file)

    all_indices = list(test_df.index)
    unused = list(set(all_indices) - set(st.session_state[f"{ds_key}_used_indices"]))

    if not unused:
        st.session_state[f"{ds_key}_used_indices"] = []
        unused = all_indices

    if st.session_state.current_question is None:
        idx = random.choice(unused)
        st.session_state[f"{ds_key}_used_indices"].append(idx)
        st.session_state.current_question = test_df.loc[idx]
        st.session_state.revealed = False

    sample = st.session_state.current_question

    st.dataframe(pd.DataFrame([sample.drop(target_col)]), use_container_width=True, hide_index=True)

    participants = st.number_input("Number of Participants", 1, 20, 1, key=f"{ds_key}_num")

    votes = []
    for i in range(participants):
        vote = st.radio(f"Participant {i+1}", vote_options, key=f"{ds_key}_{i}")
        votes.append(vote)

    if st.button("Reveal Result", key=f"{ds_key}_reveal") and not st.session_state.revealed:

        st.session_state.revealed = True

        X_sample = pd.DataFrame([sample.drop(target_col)])
        prediction = model.predict(X_sample)[0]
        actual = sample[target_col]

        pred_label = label_map[prediction]
        actual_label = label_map[actual]

        # AI score
        scores[ds_key]["ai_total"] += 1
        scores["global"]["ai_total"] += 1

        if prediction == actual:
            scores[ds_key]["ai_correct"] += 1
            scores["global"]["ai_correct"] += 1

        # Human score
        for vote in votes:
            scores[ds_key]["human_total"] += 1
            scores["global"]["human_total"] += 1
            if vote == actual_label:
                scores[ds_key]["human_correct"] += 1
                scores["global"]["human_correct"] += 1

        save_scores()

        st.success(f"AI Prediction: {pred_label}")
        st.info(f"Actual Outcome: {actual_label}")

    if st.session_state.revealed:
        if st.button("Next Group", key=f"{ds_key}_next"):
            st.session_state.current_question = None
            st.session_state.revealed = False
            st.rerun()

    # ===============================
    # DATASET SPECIFIC LEADERBOARD
    # ===============================

    st.divider()
    st.subheader("📊 Dataset Leaderboard")

    ai_correct = scores[ds_key]["ai_correct"]
    ai_total = scores[ds_key]["ai_total"]
    human_correct = scores[ds_key]["human_correct"]
    human_total = scores[ds_key]["human_total"]

    st.write(f"🤖 AI: {ai_correct} / {ai_total} ({calc_accuracy(ai_correct, ai_total):.2f}%)")
    st.write(f"👥 Humans: {human_correct} / {human_total} ({calc_accuracy(human_correct, human_total):.2f}%)")

# =====================================================
# STUDENT
# =====================================================

if dataset_choice == "Student Performance":

    st.subheader("📚 Understand the Data First")
    st.markdown('<div class="banner">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    run_dataset(
        "student",
        "student_model.pkl",
        "student_test.csv",
        "Pass_Fail",
        ["PASS", "FAIL"],
        {1: "PASS", 0: "FAIL"}
    )

# =====================================================
# HEART
# =====================================================

elif dataset_choice == "Heart Disease":

    st.subheader("❤️ Understand the Data First")
    st.markdown('<div class="banner">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    run_dataset(
        "heart",
        "heart_model.pkl",
        "heart_test.csv",
        "target",
        ["Disease", "No Disease"],
        {1: "Disease", 0: "No Disease"}
    )

# =====================================================
# PERSONALITY
# =====================================================

elif dataset_choice == "Personality Prediction":

    st.subheader("🎭 Understand the Data First")
    st.markdown('<div class="banner">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    run_dataset(
        "personality",
        "personality_model.pkl",
        "personality_test.csv",
        "Personality",
        ["Introvert", "Extrovert"],
        {1: "Introvert", 0: "Extrovert"}
    )

