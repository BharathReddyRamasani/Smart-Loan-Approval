import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    df = df.copy()

    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(0)
    df["Has_Coapplicant"] = (df["CoapplicantIncome"] > 0).astype(int)

    X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
    y = df["Loan_Status"]

    numeric_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Has_Coapplicant"
    ]

    categorical_cols = [
        "Self_Employed",
        "Property_Area"
    ]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    return X, y, preprocessor
