import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- VIP DARK THEME CSS ----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #131a17 0%, #090c0b 100%);
        color: #f5f6fa;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ffd166, #06d6a0, #00bbf9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: 1px;
    }

    .sub-title {
        text-align: center;
        color: #8b9a95;
        font-size: 1.05rem;
        margin-bottom: 30px;
        font-weight: 400;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #131a17, #1a2420);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #223028;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    div[data-testid="stMetric"] label {
        color: #8b9a95 !important;
    }

    .result-box {
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 18px;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }

    .approved-box {
        background: linear-gradient(145deg, #003d2b, #002417);
        border: 2px solid #06d6a0;
        color: #06d6a0;
    }

    .rejected-box {
        background: linear-gradient(145deg, #3d0014, #240009);
        border: 2px solid #ff4d6d;
        color: #ff4d6d;
    }

    .stButton>button {
        background: linear-gradient(90deg, #06d6a0, #00bbf9);
        color: #06110d;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 12px 0px;
        font-size: 1rem;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(6, 214, 160, 0.5);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1512, #090c0b);
        border-right: 1px solid #1c2521;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #06d6a0;
        margin-top: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #ffd166;
        padding-left: 10px;
    }

    hr {
        border-color: #1c2521;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD + TRAIN (cached) ----------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("data.csv")

    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
    df["Credit_History"].fillna(df["Credit_History"].median(), inplace=True)

    le_target = LabelEncoder()
    df["Loan_Status"] = le_target.fit_transform(df["Loan_Status"])  # N=0, Y=1

    le_married = LabelEncoder()
    df["Married"] = le_married.fit_transform(df["Married"])

    le_gender = LabelEncoder()
    df["Gender"] = le_gender.fit_transform(df["Gender"])

    le_selfemp = LabelEncoder()
    df["Self_Employed"] = le_selfemp.fit_transform(df["Self_Employed"])

    df_encoded = pd.get_dummies(df, columns=["Dependents", "Education", "Property_Area"], drop_first=True)

    x = df_encoded.drop(columns=["Loan_Status", "Loan_ID"])
    y = df_encoded["Loan_Status"]
    train_columns = x.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- CatBoost (light grid for fast app startup) ----
    cat_grid = GridSearchCV(
        CatBoostClassifier(random_state=42, verbose=0),
        {"iterations": [100, 200], "depth": [3, 5], "learning_rate": [0.05, 0.1]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    cat_grid.fit(x_train, y_train)
    cat_pred = cat_grid.predict(x_test)

    # ---- XGBoost ----
    xg_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    xg_grid.fit(x_train, y_train)
    xg_pred = xg_grid.predict(x_test)

    # ---- LightGBM ----
    gb_grid = GridSearchCV(
        LGBMClassifier(random_state=42, verbose=-1),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    gb_grid.fit(x_train, y_train)
    gb_pred = gb_grid.predict(x_test)

    results = {
        "CatBoost": {
            "model": cat_grid, "pred": cat_pred,
            "accuracy": accuracy_score(y_test, cat_pred),
            "cm": confusion_matrix(y_test, cat_pred),
            "report": classification_report(y_test, cat_pred, output_dict=True),
            "best_params": cat_grid.best_params_,
            "cv_score": cat_grid.best_score_
        },
        "XGBoost": {
            "model": xg_grid, "pred": xg_pred,
            "accuracy": accuracy_score(y_test, xg_pred),
            "cm": confusion_matrix(y_test, xg_pred),
            "report": classification_report(y_test, xg_pred, output_dict=True),
            "best_params": xg_grid.best_params_,
            "cv_score": xg_grid.best_score_
        },
        "LightGBM": {
            "model": gb_grid, "pred": gb_pred,
            "accuracy": accuracy_score(y_test, gb_pred),
            "cm": confusion_matrix(y_test, gb_pred),
            "report": classification_report(y_test, gb_pred, output_dict=True),
            "best_params": gb_grid.best_params_,
            "cv_score": gb_grid.best_score_
        }
    }

    return df, x, y, results, train_columns, le_married, le_gender, le_selfemp, le_target

df, x_full, y_full, results, train_columns, le_married, le_gender, le_selfemp, le_target = load_and_train()

# ---------------- PREPROCESSING FUNCTION FOR LIVE INPUT ----------------
def preprocess_input(raw):
    row = pd.DataFrame([raw])

    row["Married"] = le_married.transform(row["Married"])
    row["Gender"] = le_gender.transform(row["Gender"])
    row["Self_Employed"] = le_selfemp.transform(row["Self_Employed"])

    row_encoded = pd.get_dummies(row, columns=["Dependents", "Education", "Property_Area"], drop_first=True)

    # align to training columns (missing dummy columns filled with 0)
    row_final = row_encoded.reindex(columns=train_columns, fill_value=0)
    return row_final

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🏦 Loan Approval Prediction AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Multi-Model Comparison: CatBoost vs XGBoost vs LightGBM</p>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<p class="section-header">🤖 Choose Model</p>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Active model for prediction:", list(results.keys()), index=0)
    active = results[selected_model_name]

    st.divider()
    st.markdown('<p class="section-header">⚙️ Model Performance</p>', unsafe_allow_html=True)
    st.metric("Test Accuracy", f"{active['accuracy']*100:.2f}%")
    st.metric("Cross-Val Score", f"{active['cv_score']*100:.2f}%")
    with st.expander("Best Hyperparameters"):
        st.json(active["best_params"])

    st.divider()
    st.markdown('<p class="section-header">📋 Dataset Snapshot</p>', unsafe_allow_html=True)
    st.metric("Total Applicants", len(df))
    st.metric("Approved Loans", int((df["Loan_Status"] == 1).sum()))
    st.metric("Rejected Loans", int((df["Loan_Status"] == 0).sum()))

# ---------------- MODEL COMPARISON ----------------
st.markdown('<p class="section-header">📊 Model Comparison</p>', unsafe_allow_html=True)

comp_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[m]["accuracy"] * 100 for m in results],
    "CV Score": [results[m]["cv_score"] * 100 for m in results]
})

c1, c2 = st.columns([1.2, 1])
with c1:
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["Accuracy"], name="Test Accuracy", marker_color="#06d6a0"))
    fig_comp.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["CV Score"], name="CV Score", marker_color="#00bbf9"))
    fig_comp.update_layout(
        template="plotly_dark", barmode="group", height=380,
        title="Accuracy vs Cross-Validation Score",
        yaxis_title="Score (%)",
        plot_bgcolor="#090c0b", paper_bgcolor="#090c0b",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with c2:
    st.dataframe(comp_df.round(2), use_container_width=True, hide_index=True)
    best_model = comp_df.loc[comp_df["Accuracy"].idxmax(), "Model"]
    st.success(f"🏆 Best performing model: **{best_model}**")

# ---------------- LIVE PREDICTION ----------------
st.divider()
st.markdown('<p class="section-header">🔍 Check Loan Eligibility</p>', unsafe_allow_html=True)
st.caption(f"Using **{selected_model_name}** — fill in applicant details below to get an instant prediction.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Applicant Info**")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    st.markdown("**💰 Financial Info**")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=int(df["ApplicantIncome"].median()))
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=int(df["CoapplicantIncome"].median()))
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=int(df["LoanAmount"].median()))
    loan_term = st.selectbox("Loan Amount Term (days)", sorted(df["Loan_Amount_Term"].dropna().unique().tolist()), index=0)

with col3:
    st.markdown("**🏠 Other Info**")
    credit_history = st.selectbox("Credit History Meets Guidelines?", [1.0, 0.0], format_func=lambda v: "Yes" if v == 1.0 else "No")
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

predict_btn = st.button("🚀 Predict Loan Status", use_container_width=True, type="primary")

if predict_btn:
    raw_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    input_processed = preprocess_input(raw_input)
    pred = active["model"].predict(input_processed)[0]
    proba = active["model"].predict_proba(input_processed)[0]

    reject_prob = proba[0] * 100
    approve_prob = proba[1] * 100

    if pred == 1:
        st.markdown(f'<div class="result-box approved-box">✅ LOAN APPROVED ({approve_prob:.1f}% confidence)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box rejected-box">🚨 LOAN REJECTED ({reject_prob:.1f}% confidence)</div>', unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=[reject_prob, approve_prob],
        y=["Rejected", "Approved"],
        orientation="h",
        marker_color=["#ff4d6d", "#06d6a0"],
        text=[f"{reject_prob:.1f}%", f"{approve_prob:.1f}%"],
        textposition="auto"
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Prediction Confidence",
        xaxis_title="Probability (%)",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#090c0b",
        paper_bgcolor="#090c0b"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ This tool is for educational/portfolio purposes only and does not reflect real lending decisions.")

# ---------------- CONFUSION MATRIX + CLASSIFICATION REPORT ----------------
st.divider()
st.markdown('<p class="section-header">📈 Detailed Evaluation</p>', unsafe_allow_html=True)

eval_col1, eval_col2 = st.columns(2)

with eval_col1:
    st.write(f"**Confusion Matrix — {selected_model_name}**")
    fig_cm = px.imshow(
        active["cm"], text_auto=True, color_continuous_scale="Greens",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Rejected", "Approved"], y=["Rejected", "Approved"]
    )
    fig_cm.update_layout(
        template="plotly_dark", height=350,
        plot_bgcolor="#090c0b", paper_bgcolor="#090c0b",
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with eval_col2:
    st.write(f"**Classification Report — {selected_model_name}**")
    report_df = pd.DataFrame(active["report"]).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

# ---------------- EXPLORATORY VISUALS ----------------
st.divider()
st.markdown('<p class="section-header">🔬 Data Exploration</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Approval by Property Area", "💰 Income vs Loan Amount", "🎓 Approval by Education"])

with tab1:
    area_df = df.groupby("Property_Area")["Loan_Status"].mean().reset_index()
    area_df["Approval Rate (%)"] = area_df["Loan_Status"] * 100
    fig_area = px.bar(area_df, x="Property_Area", y="Approval Rate (%)", color="Property_Area",
                       color_discrete_sequence=["#06d6a0", "#00bbf9", "#ffd166"])
    fig_area.update_layout(template="plotly_dark", height=400, plot_bgcolor="#090c0b", paper_bgcolor="#090c0b")
    st.plotly_chart(fig_area, use_container_width=True)

with tab2:
    plot_df = df.copy()
    plot_df["Loan_Status_Label"] = plot_df["Loan_Status"].map({1: "Approved", 0: "Rejected"})
    fig_scatter = px.scatter(
        plot_df, x="ApplicantIncome", y="LoanAmount", color="Loan_Status_Label",
        color_discrete_map={"Approved": "#06d6a0", "Rejected": "#ff4d6d"}, opacity=0.7
    )
    fig_scatter.update_layout(template="plotly_dark", height=450, plot_bgcolor="#090c0b", paper_bgcolor="#090c0b")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    edu_df = df.groupby("Education")["Loan_Status"].mean().reset_index()
    edu_df["Approval Rate (%)"] = edu_df["Loan_Status"] * 100
    fig_edu = px.bar(edu_df, x="Education", y="Approval Rate (%)", color="Education",
                      color_discrete_sequence=["#06d6a0", "#ffd166"])
    fig_edu.update_layout(template="plotly_dark", height=400, plot_bgcolor="#090c0b", paper_bgcolor="#090c0b")
    st.plotly_chart(fig_edu, use_container_width=True)

# ---------------- RAW DATA ----------------
with st.expander("📋 View Raw Dataset Sample"):
    st.dataframe(df.sample(10, random_state=1), use_container_width=True)
