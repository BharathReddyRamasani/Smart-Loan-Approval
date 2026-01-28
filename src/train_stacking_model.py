import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from preprocessing import preprocess_data

df = pd.read_csv("data/train.csv")
X, y, preprocessor = preprocess_data(df)

base_models = [
    ("lr", LogisticRegression(max_iter=1000)),
    ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance")),
    ("svm", SVC(kernel="rbf", probability=True)),
    ("dt", DecisionTreeClassifier(max_depth=5)),
    ("rf", RandomForestClassifier(n_estimators=200)),
    ("xgb", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    ))
]

meta_model = LogisticRegression(max_iter=1000)

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

stack_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("stack", stack)
])

stack_pipe.fit(X, y)
joblib.dump(stack_pipe, "models/stacking_model.pkl")

print("✅ Stacking model trained and saved")
