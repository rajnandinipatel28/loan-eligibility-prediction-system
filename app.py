import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load model
# -------------------------------
model = joblib.load("model/loan_eligibility_model.pkl")

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Loan Eligibility Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Eligibility Prediction")
st.write("Enter applicant details to check loan approval status")

# -------------------------------
# User Inputs
# -------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
app_income = st.number_input("Applicant Income", min_value=0)
coapp_income = st.number_input("Co-applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.number_input("Loan Amount Term (months)", value=360)
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------
# Encode inputs (same logic as training)
# -------------------------------
def encode_inputs():
    return np.array([[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        3 if dependents == "3+" else int(dependents),
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        app_income,
        coapp_income,
        loan_amount,
        loan_term,
        1 if credit_history == "Good (1)" else 0,
        {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    ]])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Check Loan Eligibility"):
    input_data = encode_inputs()
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved (Confidence: {probability*100:.2f}%)")
    else:
        st.error(f"❌ Loan Not Approved (Confidence: {(1-probability)*100:.2f}%)")