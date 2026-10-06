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

    # Some mirrors of this dataset ship with the categorical columns already
    # numerically encoded (e.g. via sklearn's LabelEncoder, which assigns
    # integers in alphabetical order of the original category names) instead
    # of the original text labels (M/F, ATA/NAP/ASY/TA, Normal/ST/LVH, Y/N).
    # Logistic Regression in particular is sensitive to this, since it
    # treats the numbers as having numeric order/magnitude rather than as
    # unordered categories, which silently distorts its predictions. If a
    # column is found to be purely numeric and its values exactly match this
    # known encoding, it is decoded back to the original text labels before
    # training, so results are consistent regardless of which version of the
    # dataset is used.
    LABEL_ENCODE_MAPS = {
        "Sex": {0: "F", 1: "M"},
        "ChestPainType": {0: "ASY", 1: "ATA", 2: "NAP", 3: "TA"},
        "RestingECG": {0: "LVH", 1: "Normal", 2: "ST"},
        "ExerciseAngina": {0: "N", 1: "Y"},
    }
    decoded_columns = []
    for col, mapping in LABEL_ENCODE_MAPS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = set(df[col].dropna().unique())
            if unique_vals and unique_vals.issubset(set(mapping.keys())):
                df[col] = df[col].map(mapping)
                decoded_columns.append(col)

    # Remove duplicate patient records before splitting. Combined heart disease
    # datasets (merged from multiple clinical sources) often contain exact
    # duplicate rows; if a duplicate ends up in both the training and test
    # sets, a model can "memorise" it rather than genuinely generalise,
    # producing an inflated, untrustworthy score (most visible as Decision
    # Tree scoring unrealistically close to 100%).
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_duplicates_removed = n_before - len(df)

    # Ensure the target is clean binary (0 = no heart disease, 1 = heart
    # disease present). Some versions of this dataset (e.g. the raw UCI/
    # Cleveland source) use a 0-4 severity scale instead of clean 0/1. If
    # left as-is, scikit-learn would silently treat this as a 5-class
    # problem, which breaks predict_proba and every metric computed below
    # in ways that look like random, inconsistent model performance.
    raw_target_values = sorted(df["HeartDisease"].dropna().unique().tolist())
    target_was_binarized = not set(raw_target_values).issubset({0, 1})
    if target_was_binarized:
        df["HeartDisease"] = (df["HeartDisease"] > 0).astype(int)

    # Make sure every numeric column is actually numeric. If a column
    # contains a stray non-numeric marker (e.g. "?", used by some raw UCI
    # exports for missing values), pandas silently reads the whole column
    # as text, which breaks StandardScaler for Logistic Regression while
    # tree-based models may partially tolerate it -- producing exactly the
    # kind of lopsided results (one model collapsing, others fine) this
    # dataset has produced before.
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    n_missing_before_fill = df[NUMERIC_COLS].isna().sum().sum()
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"].astype(int)

    diagnostics = {
        "n_rows": len(df),
        "raw_target_values": raw_target_values,
        "target_was_binarized": target_was_binarized,
        "n_missing_numeric_values_filled": int(n_missing_before_fill),
        "numeric_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC_COLS},
        "categorical_values": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_COLS},
    }

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
    return preprocessor, final_base_models, meta_model, metrics_df, n_duplicates_removed, decoded_columns, diagnostics


(preprocessor, base_models, meta_model, metrics_df,
 n_duplicates_removed, decoded_columns, diagnostics) = train_all_models()
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

    if n_duplicates_removed > 0:
        st.info(
            f"ℹ️ {n_duplicates_removed} duplicate patient record(s) were found in the dataset and "
            "removed before training. This prevents a model from simply memorising a record it has "
            "already seen if that same record also appears in the test set -- which would otherwise "
            "produce an unrealistically high, untrustworthy score (most noticeable as Decision Tree "
            "scoring close to 100%)."
        )

    if decoded_columns:
        st.info(
            f"ℹ️ The following column(s) were found in a numerically-encoded form in the uploaded "
            f"dataset and were automatically converted back to their original category labels "
            f"before training, to keep results consistent: {', '.join(decoded_columns)}."
        )

    if diagnostics["target_was_binarized"]:
        st.warning(
            f"⚠️ The HeartDisease column in the uploaded dataset contained values "
            f"{diagnostics['raw_target_values']}, not clean 0/1. It was automatically converted "
            f"so that any value greater than 0 is treated as 'heart disease present' (1) and 0 "
            f"is treated as 'no heart disease' (0), consistent with Chapter Three."
        )

    if diagnostics["n_missing_numeric_values_filled"] > 0:
        st.warning(
            f"⚠️ {diagnostics['n_missing_numeric_values_filled']} missing or non-numeric value(s) "
            "were found in the numeric columns and filled with that column's median value."
        )

    with st.expander("🔍 Dataset diagnostics (for checking data quality)"):
        st.write(f"Rows used for training/testing: **{diagnostics['n_rows']}**")
        st.write("Numeric column ranges (min, max):")
        st.json({k: v for k, v in diagnostics["numeric_ranges"].items()})
        st.write("Categorical column values found:")
        st.json(diagnostics["categorical_values"])

    with st.expander("What do Accuracy, Precision, Recall, F1-score and ROC-AUC mean?"):
        st.markdown(
            "- **Accuracy**: the proportion of all predictions that were correct.\n"
            "- **Precision**: of everyone predicted to have heart disease, how many actually did.\n"
            "- **Recall**: of everyone who actually had heart disease, how many the model correctly identified. "
            "(A high recall matters most, since missing a real case is more serious than a false alarm.)\n"
            "- **F1-score**: a single score that balances Precision and Recall.\n"
            "- **ROC-AUC**: how well the model distinguishes between people with and without heart disease overall."
        )
