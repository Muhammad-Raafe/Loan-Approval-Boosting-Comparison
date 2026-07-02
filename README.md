# 🏦 Loan Approval Prediction AI

An interactive machine learning web app that predicts whether a loan application will be **Approved** or **Rejected**, comparing three powerful gradient boosting algorithms — CatBoost, XGBoost, and LightGBM — each hyperparameter-tuned with GridSearchCV. Built with Scikit-learn and deployed with Streamlit.

🔗 **Live Demo:** https://loan-approval-boosting-comparison-raafe.streamlit.app/

⚠️ **Disclaimer:** This project is for educational and portfolio purposes only and does not reflect real lending or credit decisions.

---

## 🧠 Overview

This project predicts loan approval outcomes based on applicant demographics, income, and credit history. Three state-of-the-art gradient boosting classifiers are trained, tuned, and compared side-by-side — letting the user pick which model powers the live eligibility check, and see exactly how each one performs.

---

## ✨ Features

- 🤖 **Multi-model comparison** — CatBoost, XGBoost, and LightGBM trained and tuned in parallel, all visible in one dashboard
- 🔍 **Live loan eligibility checker** — enter applicant details and get an instant Approved/Rejected prediction with confidence score
- ⚙️ **Hyperparameter tuning transparency** — view the best parameters found via GridSearchCV for each model
- 📊 **Full evaluation suite** — accuracy, cross-validation score, confusion matrix, and classification report per model
- 📈 **Interactive Plotly visualizations** — model comparison bar chart, approval rate by property area, income vs loan amount scatter, and approval rate by education
- 🎨 **Premium dark-themed UI** with gradient accents and smooth interactions
- ⚡ **Cached model training** for fast app performance

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML Library | Scikit-learn |
| Models | CatBoost, XGBoost, LightGBM |
| Hyperparameter Tuning | GridSearchCV (5-fold CV) |
| Preprocessing | LabelEncoder, One-Hot Encoding (pd.get_dummies) |
| Visualization | Plotly, Seaborn, Matplotlib |
| Web Framework | Streamlit |
| Data Handling | Pandas, NumPy |

---

## 📂 Dataset

The model is trained on a standard **Loan Prediction dataset**, containing applicant demographic and financial information.

**Features:**
- `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
- `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`
- `Credit_History`, `Property_Area`

**Target:**
- `Loan_Status` — Y (Approved) or N (Rejected), label-encoded to 1/0

---

## ⚙️ How It Works

1. **Data Cleaning** — Filled missing values: mode imputation for categorical columns (`Dependents`, `Gender`, `Married`, `Self_Employed`), median imputation for numeric columns (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`)
2. **Encoding** — Label-encoded binary categorical columns (`Married`, `Gender`, `Self_Employed`, `Loan_Status`), one-hot encoded multi-category columns (`Dependents`, `Education`, `Property_Area`) with `drop_first=True` to avoid multicollinearity
3. **Train/Test Split** — 80/20 stratified split to preserve class balance
4. **Model 1 — CatBoost** — Tuned `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `random_strength` via GridSearchCV
5. **Model 2 — XGBoost** — Tuned `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma` via GridSearchCV
6. **Model 3 — LightGBM** — Tuned `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `num_leaves` via GridSearchCV
7. **Evaluation** — Compared all three on accuracy, cross-validation score, confusion matrix, and classification report
8. **Deployment** — Wrapped in an interactive Streamlit app where the user can pick any of the three models for live eligibility predictions, with raw form inputs automatically preprocessed to match the training pipeline

---

## 🚀 Running Locally

1. Clone the repository
```bash
git clone https://github.com/Muhammad-Raafe/Loan-Approval-Prediction-ML.git
cd Loan-Approval-Prediction-ML
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

Make sure `data.csv` is present in the project root directory.

---

## 📁 Project Structure

```
Loan-Approval-Prediction-ML/
│
├── app.py                # Streamlit web app (multi-model comparison + live prediction)
├── model_training.py      # Original model training & comparison script
├── data.csv                # Dataset
├── requirements.txt         # Python dependencies
└── README.md                 # Project documentation
```

---

## 📊 Model Performance

All three gradient boosting models are evaluated on the same held-out, stratified test set, with accuracy, cross-validation score, confusion matrix, and full classification report displayed live in the app — letting you directly compare CatBoost vs XGBoost vs LightGBM performance on this dataset.

---

## 🔮 Future Improvements

- Add SHAP-based feature importance for individual predictions
- Add ROC curve and AUC score comparison across all three models
- Support batch CSV upload for bulk loan eligibility screening
- Add fairness/bias analysis across gender and property area segments

---

## 👤 Author

**Muhammad Raafe**
AI/ML Enthusiast | Building a portfolio in Machine Learning & Data Science

GitHub: [@Muhammad-Raafe](https://github.com/Muhammad-Raafe)
