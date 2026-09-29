"""
Heart Disease Prediction System
Development, Evaluation and Comparative Analysis of Heart Disease Prediction
Using Ensemble Techniques

A working prediction system implementing the methodology from Chapter 3:
Logistic Regression, Decision Tree and XGBoost as base models, combined via
Averaging, Majority Voting and Stacking (Logistic Regression meta-model,
trained on out-of-fold predictions).

Deploy on Streamlit Community Cloud (share.streamlit.io) -- no local
computer needed after deployment. Requires 'heart.csv' in the same repo.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "heart.csv"  # must be uploaded to the same GitHub repo as this file

NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina"]

st.set_page_config(page_title="Heart Disease Prediction System", page_icon="❤️", layout="centered")


# ---------------------------------------------------------------------------
# Training (runs once, cached for the life of the deployed app)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models on the dataset (first load only)...")
def train_all_models():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    def make_base_models():
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        }

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    base_models = make_base_models()
    metrics = []
    base_test_probs = {}
    for name, model in base_models.items():
        model.fit(X_train_t, y_train)
        prob = model.predict_proba(X_test_t)[:, 1]
        pred = (prob >= 0.5).astype(int)
        base_test_probs[name] = prob
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
        })

    # Averaging
    avg_prob = np.mean(list(base_test_probs.values()), axis=0)
    avg_pred = (avg_prob >= 0.5).astype(int)
    metrics.append({
        "Model": "Averaging",
        "Accuracy": accuracy_score(y_test, avg_pred),
        "Precision": precision_score(y_test, avg_pred, zero_division=0),
        "Recall": recall_score(y_test, avg_pred, zero_division=0),
        "F1-score": f1_score(y_test, avg_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, avg_prob),
    })

    # Majority Voting
    votes = np.array([(p >= 0.5).astype(int) for p in base_test_probs.values()])
    vote_pred = (votes.sum(axis=0) >= 2).astype(int)
    vote_prob = votes.mean(axis=0)
    metrics.append({
        "Model": "Majority Voting",
        "Accuracy": accuracy_score(y_test, vote_pred),
        "Precision": precision_score(y_test, vote_pred, zero_division=0),
        "Recall": recall_score(y_test, vote_pred, zero_division=0),
        "F1-score": f1_score(y_test, vote_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, vote_prob),
    })

    # Stacking: out-of-fold predictions to train the meta-model (no leakage)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = {name: np.zeros(len(X_train)) for name in base_models}
    for name in base_models:
        for tr_idx, val_idx in skf.split(X_train_t, y_train):
            m = make_base_models()[name]
            m.fit(X_train_t[tr_idx], y_train.iloc[tr_idx])
            oof[name][val_idx] = m.predict_proba(X_train_t[val_idx])[:, 1]
    oof_df = pd.DataFrame(oof)
    meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    meta_model.fit(oof_df, y_train)

    meta_test_input = pd.DataFrame(base_test_probs)
    stack_prob = meta_model.predict_proba(meta_test_input)[:, 1]
    stack_pred = (stack_prob >= 0.5).astype(int)
    metrics.append({
        "Model": "Stacking",
        "Accuracy": accuracy_score(y_test, stack_pred),
        "Precision": precision_score(y_test, stack_pred, zero_division=0),
        "Recall": recall_score(y_test, stack_pred, zero_division=0),
        "F1-score": f1_score(y_test, stack_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, stack_prob),
    })

    # Final base models refit on ALL data (train+test) for real-world single-patient prediction
    X_all_t = preprocessor.fit_transform(X)
    final_base_models = make_base_models()
    for name, model in final_base_models.items():
        model.fit(X_all_t, y)

    metrics_df = pd.DataFrame(metrics).sort_values("F1-score", ascending=False).reset_index(drop=True)
    return preprocessor, final_base_models, meta_model, metrics_df


preprocessor, base_models, meta_model, metrics_df = train_all_models()
best_model_name = metrics_df.iloc[0]["Model"]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction System")
st.caption(
    "Development, Evaluation and Comparative Analysis of Heart Disease Prediction "
    "Using Ensemble Techniques -- Logistic Regression, Decision Tree and XGBoost, "
    "combined via Averaging, Majority Voting and Stacking."
)

tab1, tab2 = st.tabs(["🩺 Predict", "📊 Model Performance"])

with tab1:
    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox(
            "Chest Pain Type", ["ATA", "NAP", "ASY", "TA"],
            help="TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic",
        )
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

    with col2:
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)

    st.divider()

    show_all = st.checkbox("Show predictions from all six models (not just the best one)")

    if st.button("Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": 1 if fasting_bs == "Yes" else 0,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
        }])

        input_t = preprocessor.transform(input_df)
        base_probs = {name: model.predict_proba(input_t)[:, 1][0] for name, model in base_models.items()}

        avg_prob = np.mean(list(base_probs.values()))
        votes = [1 if p >= 0.5 else 0 for p in base_probs.values()]
        vote_prob = np.mean(votes)
        stack_input = pd.DataFrame([base_probs])
        stack_prob = meta_model.predict_proba(stack_input)[:, 1][0]

        all_results = {
            **base_probs,
            "Averaging": avg_prob,
            "Majority Voting": vote_prob,
            "Stacking": stack_prob,
        }

        best_prob = all_results[best_model_name]
        best_pred = "Heart Disease Likely" if best_prob >= 0.5 else "Heart Disease Unlikely"

        st.subheader("Result")
        if best_prob >= 0.5:
            st.error(f"**{best_pred}** (using {best_model_name} — the best-performing model)")
        else:
            st.success(f"**{best_pred}** (using {best_model_name} — the best-performing model)")
        st.metric("Predicted probability of heart disease", f"{best_prob*100:.1f}%")
        st.caption(
            "This is a prediction from a machine learning model trained on historical data, "
            "not a medical diagnosis. Consult a qualified clinician for actual medical decisions."
        )

        if show_all:
            st.divider()
            st.subheader("All Six Models")
            comp_df = pd.DataFrame({
                "Model": list(all_results.keys()),
                "Predicted Probability": [f"{v*100:.1f}%" for v in all_results.values()],
                "Predicted Class": ["Disease Likely" if v >= 0.5 else "Disease Unlikely" for v in all_results.values()],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Performance on the Held-Out Test Set")
    st.caption("All six models evaluated on the same 20% test split, as described in Chapter 3.")
    display_df = metrics_df.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
        display_df[c] = display_df[c].map(lambda x: f"{x:.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(f"Best-performing model on this run: **{best_model_name}**")
