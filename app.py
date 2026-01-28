import streamlit as st
import pandas as pd

from src.predict import predict_all
from src.utils import business_explanation

st.set_page_config(page_title="Smart Loan Approval", layout="centered")

st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble Machine Learning model** "
    "to predict loan eligibility."
)

st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amt = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)
credit = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

credit_val = 1.0 if credit == "Yes" else 0.0
has_co = 1 if co_income > 0 else 0

input_df = pd.DataFrame([{
    "ApplicantIncome": app_income,
    "CoapplicantIncome": co_income,
    "LoanAmount": loan_amt,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_val,
    "Self_Employed": "Yes" if employment == "Self-Employed" else "No",
    "Property_Area": property_area,
    "Has_Coapplicant": has_co
}])

st.markdown("### 🧠 Stacking Architecture")
st.markdown("""
**Base Models**
- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest
- XGBoost

**Meta Model**
- Logistic Regression
""")

if st.button("🔘 Check Loan Eligibility (Stacking Model)"):
    base_preds, final_pred, confidence = predict_all(input_df)

    st.markdown("## 📊 Base Model Predictions")
    for model, pred in base_preds.items():
        st.write(f"{model} → {'Approved' if pred == 1 else 'Rejected'}")

    st.markdown("## 🧠 Final Stacking Decision")
    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"📈 Confidence Score: **{confidence}%**")

    st.markdown("## 🏦 Business Explanation")
    st.info(business_explanation(final_pred))
