import joblib

lr = joblib.load("models/base_lr.pkl")
knn = joblib.load("models/base_knn.pkl")
svm = joblib.load("models/base_svm.pkl")
dt = joblib.load("models/base_dt.pkl")
rf = joblib.load("models/base_rf.pkl")
xgb = joblib.load("models/base_xgb.pkl")
stack = joblib.load("models/stacking_model.pkl")

def predict_all(input_df):
    base_preds = {
        "Logistic Regression": lr.predict(input_df)[0],
        "KNN": knn.predict(input_df)[0],
        "SVM": svm.predict(input_df)[0],
        "Decision Tree": dt.predict(input_df)[0],
        "Random Forest": rf.predict(input_df)[0],
        "XGBoost": xgb.predict(input_df)[0]
    }

    final_pred = stack.predict(input_df)[0]
    confidence = stack.predict_proba(input_df)[0][final_pred]

    return base_preds, final_pred, round(confidence * 100, 2)
