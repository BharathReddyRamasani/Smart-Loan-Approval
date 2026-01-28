import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from preprocessing import preprocess_data

df = pd.read_csv("data/train.csv")
X, y, preprocessor = preprocess_data(df)

models = {
    "base_lr": LogisticRegression(max_iter=1000),

    "base_knn": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    ),

    "base_svm": SVC(
        kernel="rbf",
        probability=True
    ),

    "base_dt": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "base_rf": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "base_xgb": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, f"models/{name}.pkl")
    print(f"✅ {name} trained and saved")
