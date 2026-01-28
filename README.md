🎯 Smart Loan Approval System – Stacking Ensemble Model

A professional loan eligibility prediction system built using a Stacking Ensemble Machine Learning approach and deployed with Streamlit.

This system predicts whether a loan will be Approved or Rejected by combining predictions from multiple machine learning models, ensuring better accuracy, robustness, and explainability.

📌 Problem Statement

Banks and financial institutions must decide whether a loan applicant is likely to repay a loan based on financial and demographic information.

Traditional single models may fail to capture complex patterns.
To overcome this, we use a Stacking Ensemble, where multiple models collaborate to make a final decision.

🧠 Solution Approach

We use a Stacking Ensemble Learning Architecture:

🔹 Base Models

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

XGBoost (optional / environment-dependent)

🔹 Meta Model

Logistic Regression

The meta-model learns from the predictions of base models and produces the final loan approval decision.

🏗️ Model Architecture
User Input
   ↓
Data Preprocessing
   ↓
Base Models
   ├── Logistic Regression
   ├── KNN
   ├── SVM
   ├── Decision Tree
   ├── Random Forest
   └── XGBoost (optional)
   ↓
Meta Model (Logistic Regression)
   ↓
Final Loan Approval / Rejection

📊 Dataset Description

The dataset contains historical loan application records with the following key features:

Feature	Description
ApplicantIncome	Income of the primary applicant
CoapplicantIncome	Income of co-applicant (0 if none)
LoanAmount	Loan amount requested
Loan_Amount_Term	Loan tenure (months)
Credit_History	Credit repayment history
Self_Employed	Employment type
Property_Area	Property location
Loan_Status	Target variable (Y / N)

📌 Note:
Missing values are handled using domain-aware preprocessing.

⚙️ Data Preprocessing (Important)

CoapplicantIncome
→ Missing value means no co-applicant, so filled with 0

Has_Coapplicant
→ New binary feature added

Numerical missing values
→ Median imputation

Categorical missing values
→ Filled with "UNKNOWN"

Scaling
→ Applied using StandardScaler

All preprocessing is done using Scikit-learn Pipelines to avoid data leakage.

🎨 User Interface (Streamlit)
Key UI Features:

Sidebar-based input form

Clean, professional, high-contrast UI

Base model predictions shown individually

Final stacking decision highlighted

Confidence score displayed

Business-friendly explanation provided

Sample Output:

✅ Loan Approved

❌ Loan Rejected

🏦 Business Explanation (Explainability)

The system explains decisions in simple terms:

“Based on income, credit history, and combined predictions from multiple machine learning models, the applicant is likely / unlikely to repay the loan. Therefore, the loan is approved / rejected.”

This ensures the model is interpretable for non-technical stakeholders.

🧰 Tech Stack

Language: Python

Machine Learning: Scikit-learn, XGBoost

Web Framework: Streamlit

Data Handling: Pandas, NumPy

Model Saving: Joblib

📁 Project Structure
smart-loan-approval/
│
├── app.py                  # Streamlit UI
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
│
├── data/
│   └── train.csv           # Dataset
│
├── models/
│   ├── base_lr.pkl
│   ├── base_knn.pkl
│   ├── base_svm.pkl
│   ├── base_dt.pkl
│   ├── base_rf.pkl
│   ├── base_xgb.pkl
│   └── stacking_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_base_models.py
│   ├── train_stacking_model.py
│   ├── predict.py
│   └── utils.py

▶️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train base models
python src/train_base_models.py

3️⃣ Train stacking model
python src/train_stacking_model.py

4️⃣ Run Streamlit app
streamlit run app.py

🌐 Deployment Notes (Streamlit Cloud)

XGBoost may not be supported on all cloud environments

A fallback mechanism is implemented so the app runs even if XGBoost is unavailable

This ensures high reliability and zero crashes

🎓 Academic & Resume Value

✔ Demonstrates ensemble learning
✔ Uses industry-standard pipelines
✔ Explainable ML system
✔ End-to-end deployment
✔ Suitable for placements, hackathons, and viva

🚀 Future Enhancements

SHAP-based model explanations

Model accuracy comparison dashboard

Light / Dark theme toggle

API-based deployment

Mobile-responsive UI

👨‍💻 Author

Smart Loan Approval System
Built using Stacking Ensemble Machine Learning and Streamlit.
