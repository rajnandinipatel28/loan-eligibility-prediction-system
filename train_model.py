import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("dataset/loan_data.csv")

# -------------------------------
# Data Cleaning
# -------------------------------
df.fillna({
    "Gender": df["Gender"].mode()[0],
    "Married": df["Married"].mode()[0],
    "Dependents": df["Dependents"].mode()[0],
    "Self_Employed": df["Self_Employed"].mode()[0],
    "LoanAmount": df["LoanAmount"].median(),
    "Loan_Amount_Term": df["Loan_Amount_Term"].mode()[0],
    "Credit_History": df["Credit_History"].mode()[0]
}, inplace=True)

# Drop Loan_ID (not useful)
df.drop("Loan_ID", axis=1, inplace=True)

# -------------------------------
# Encode categorical columns
# -------------------------------
label_cols = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed",
    "Property_Area", "Loan_Status"
]

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# Split features & target
# -------------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Pipeline (Scaling + Model)
# -------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

# -------------------------------
# Train model
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# Save model
# -------------------------------
joblib.dump(pipeline, "model/loan_eligibility_model.pkl")
print("Model saved successfully!")