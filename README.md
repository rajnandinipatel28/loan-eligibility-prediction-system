# 🏦 Loan Eligibility Prediction System

A machine learning-based web application that predicts whether a loan application should be approved or rejected, built using **Logistic Regression** and deployed via **Streamlit**.

---

## 📌 Project Overview

Banks and financial institutions process thousands of loan applications daily. Traditional approval processes rely on manual verification and rule-based decisions, which are time-consuming, inconsistent, and prone to human bias.

This project automates the loan eligibility assessment process using a trained ML model that evaluates applicant data and provides instant, accurate predictions.

> Presented by **Rajnandini Patel** | CSE Department, Lakshmi Narain College of Technology, Bhopal  
> Part of the **Microsoft Elevate** program

---

## 🚀 Features

- Predicts loan approval/rejection in real time
- Web-based interactive dashboard (Streamlit)
- Displays prediction confidence level (e.g., 100%)
- Handles multiple input parameters for accurate assessment
- Reduces manual effort and minimizes bias in decisions

---

## 🧠 Algorithm

**Logistic Regression** is used for binary classification:
- Output: `Approved` or `Not Approved`
- Estimates the probability of loan approval based on input features
- Well-suited for structured, tabular datasets

---

## 📥 Input Parameters

| Parameter              | Description                          |
|------------------------|--------------------------------------|
| Applicant Income       | Monthly income of the applicant      |
| Co-applicant Income    | Monthly income of the co-applicant   |
| Loan Amount            | Requested loan amount (in thousands) |
| Loan Amount Term       | Duration of the loan (in months)     |
| Credit History         | Good (1) or Bad (0)                  |
| Education              | Graduate / Not Graduate              |
| Self Employed Status   | Yes / No                             |
| Property Area          | Urban / Semi-Urban / Rural           |

---

## 🛠️ System Approach

1. **Data Collection** — Historical loan dataset with applicant details and approval status
2. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables (education, employment, property area)
   - Normalizing numerical features (income, loan amount)
3. **Feature Selection** — Selecting most relevant features for model accuracy
4. **Model Training & Evaluation** — Train/test split for performance evaluation
5. **Deployment** — Streamlit-based local web interface for real-time predictions

---

## 💻 Tech Stack

- **Language:** Python
- **ML Library:** Scikit-learn
- **Web Framework:** Streamlit
- **Dataset:** Kaggle Loan Prediction Dataset

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/rajnandinipatel028/loan-eligibility-prediction-system.git
cd loan-eligibility-prediction-system

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Then open your browser at `http://localhost:8502`

---

## 📊 Result

The dashboard accepts user inputs and outputs:
- ✅ **Loan Approved** — with confidence percentage
- ❌ **Loan Not Approved** — with confidence percentage (e.g., Confidence: 100.00%)

---

## 🔮 Future Scope

- Implement advanced algorithms: Random Forest, Gradient Boosting, XGBoost
- Integrate real-time credit score and banking data APIs
- Deploy on cloud platforms (AWS / Azure / GCP) for large-scale usage
- Add Explainable AI (XAI) features to justify decisions
- Extend support for home, education, and business loan types

---

## 📚 References

- [Scikit-learn Official Documentation](https://scikit-learn.org)
- [Kaggle Loan Prediction Dataset](https://www.kaggle.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- Research papers on ML-based credit risk assessment

---

## 🔗 GitHub Repository

[https://github.com/rajnandinipatel028/loan-eligibility-prediction-system](https://github.com/rajnandinipatel028/loan-eligibility-prediction-system)

---

## 📧 Contact

**Rajnandini Patel**  
Email: rajnandiniatel2007@gmail.com  
College: Lakshmi Narain College of Technology, Bhopal
