"""
Heart Disease Prediction System
Development, Evaluation and Comparative Analysis of Heart Disease Prediction
Using Ensemble Techniques

A working prediction system implementing the methodology from Chapter 3:
Logistic Regression, Decision Tree and XGBoost as base models, combined via
Averaging, Majority Voting and Stacking (Logistic Regression meta-model,
trained on out-of-fold predictions).

This version uses plain, patient-friendly language in the interface -- no
clinical abbreviations are shown to the user. Friendly selections are mapped
internally to the exact category codes the trained models expect.

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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "heart.csv"  # must be uploaded to the same GitHub repo as this file

NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina"]

st.set_page_config(page_title="Heart Health Checker", page_icon="❤️", layout="centered")


# ---------------------------------------------------------------------------
# Plain-language option mappings
# (friendly label shown to the user) -> (exact code the trained model expects)
# ---------------------------------------------------------------------------
SEX_OPTIONS = {"Male": "M", "Female": "F"}

CHEST_PAIN_OPTIONS = {
    "No chest pain": "ASY",
    "Typical heart-related chest pain": "TA",
    "Chest pain that doesn't clearly seem heart-related": "ATA",
    "Chest pain not related to the heart": "NAP",
}

RESTING_ECG_OPTIONS = {
    "Normal": "Normal",
    "Minor irregularity in heart rhythm (ST-T wave changes)": "ST",
    "Signs of a thickened heart muscle (left ventricular hypertrophy)": "LVH",
}

YES_NO = {"No": 0, "Yes": 1}
YES_NO_LETTER = {"No": "N", "Yes": "Y"}


# ---------------------------------------------------------------------------
# Training (runs once, cached for the life of the deployed app)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting things up (first load only, about a minute)...")
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
st.title("❤️ Heart Health Checker")
st.caption(
    "This tool estimates the likelihood of heart disease from a few health details. "
    "It is a demonstration of a research project and is not a medical diagnosis."
)

tab1, tab2 = st.tabs(["🩺 Check My Risk", "📊 How Accurate Is This Tool?"])

with tab1:
    st.subheader("About You")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Your age (years)", min_value=1, max_value=120, value=50)
    with col2:
        sex_label = st.selectbox("Sex", list(SEX_OPTIONS.keys()))

    st.subheader("Chest Pain")
    chest_pain_label = st.selectbox(
        "Which best describes any chest pain you experience?",
        list(CHEST_PAIN_OPTIONS.keys()),
        help=(
            "'Typical' chest pain feels like classic angina -- tightness or pressure brought on "
            "by exertion. 'Atypical' means it doesn't fully match that pattern. 'Not related to "
            "the heart' means the pain has another likely cause (e.g. muscular). Choose 'No "
            "chest pain' if you don't experience any."
        ),
    )
    exercise_angina_label = st.radio(
        "Do you get chest pain or tightness during physical activity or exercise?",
        list(YES_NO.keys()), horizontal=True,
    )

    st.subheader("Blood Pressure & Cholesterol")
    col3, col4 = st.columns(2)
    with col3:
        resting_bp = st.number_input(
            "Resting blood pressure (mm Hg)", min_value=0, max_value=250, value=120,
            help="The top number from a blood pressure reading, taken at rest. A typical healthy value is around 120.",
        )
    with col4:
        cholesterol = st.number_input(
            "Cholesterol level (mg/dL)", min_value=0, max_value=700, value=200,
            help="From a blood test. A commonly cited healthy level is under 200 mg/dL.",
        )
    fasting_bs_label = st.radio(
        "Was your blood sugar above 120 mg/dL after fasting (not eating for several hours)?",
        list(YES_NO.keys()), horizontal=True,
        help="This is usually measured by a doctor or lab as part of a fasting blood sugar test.",
    )

    st.subheader("Heart Test Results")
    st.caption("These come from an ECG (electrocardiogram) or exercise stress test. If you have a recent report, use the values from it. Otherwise, the default values are reasonable typical values.")
    col5, col6 = st.columns(2)
    with col5:
        resting_ecg_label = st.selectbox(
            "Resting ECG (heart rhythm test) result", list(RESTING_ECG_OPTIONS.keys())
        )
    with col6:
        max_hr = st.number_input(
            "Highest heart rate reached during an exercise test (beats per minute)",
            min_value=60, max_value=220, value=150,
        )
    oldpeak = st.slider(
        "Heart stress test score (ST depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1,
        help=(
            "A number from an exercise ECG stress test that shows how much the heart's electrical "
            "signal dips during exercise compared to rest. Higher values can indicate reduced blood "
            "flow to the heart. This number comes from a stress test report -- leave at the default "
            "if you don't have one."
        ),
    )

    st.divider()
    show_all = st.checkbox("Show results from all six models (advanced / for researchers)")

    if st.button("Check My Risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": SEX_OPTIONS[sex_label],
            "ChestPainType": CHEST_PAIN_OPTIONS[chest_pain_label],
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": YES_NO[fasting_bs_label],
            "RestingECG": RESTING_ECG_OPTIONS[resting_ecg_label],
            "MaxHR": max_hr,
            "ExerciseAngina": YES_NO_LETTER[exercise_angina_label],
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

        st.subheader("Result")
        if best_prob >= 0.5:
            st.error("**Higher likelihood of heart disease detected.**")
        else:
            st.success("**Lower likelihood of heart disease detected.**")
        st.metric("Estimated likelihood of heart disease", f"{best_prob*100:.1f}%")
        st.caption(
            "This is an estimate from a machine learning model trained on historical data, "
            "not a medical diagnosis. Please see a doctor for an accurate assessment, "
            "especially if this result concerns you."
        )

        if show_all:
            st.divider()
            st.subheader("All Six Models (Technical Comparison)")
            comp_df = pd.DataFrame({
                "Model": list(all_results.keys()),
                "Estimated Likelihood": [f"{v*100:.1f}%" for v in all_results.values()],
                "Result": ["Higher likelihood" if v >= 0.5 else "Lower likelihood" for v in all_results.values()],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("How Well Does This Tool Perform?")
    st.caption(
        "These figures show how accurately each of the six models predicted heart disease on "
        "patients the models had not seen before, during testing."
    )
    display_df = metrics_df.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
        display_df[c] = display_df[c].map(lambda x: f"{x:.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(f"Best-performing model on this run: **{best_model_name}** -- this is the model used for your result above.")

    with st.expander("What do Accuracy, Precision, Recall, F1-score and ROC-AUC mean?"):
        st.markdown(
            "- **Accuracy**: the proportion of all predictions that were correct.\n"
            "- **Precision**: of everyone predicted to have heart disease, how many actually did.\n"
            "- **Recall**: of everyone who actually had heart disease, how many the model correctly identified. "
            "(A high recall matters most, since missing a real case is more serious than a false alarm.)\n"
            "- **F1-score**: a single score that balances Precision and Recall.\n"
            "- **ROC-AUC**: how well the model distinguishes between people with and without heart disease overall."
        )
