
# MSLearn-Customer-Churn-Prediction
Predicting customer churn using ML (Telco dataset) — MS Learn project.

This project is part of Microsoft Learn’s AI/ML module. It focuses on predicting customer churn using classical machine learning algorithms like Decision Trees and XGBoost, along with data preprocessing, SMOTE for imbalance handling, and business insight extraction.

## 2. Dataset

The dataset used for this project is the **Telco Customer Churn dataset**, publicly available on Kaggle. It includes customer demographics, services subscribed, and account information.

* **Source:** [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Key Features:**
    * `gender`, `SeniorCitizen`, `Partner`, `Dependents` (Demographics)
    * `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` (Services subscribed)
    * `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` (Account Information)
    * `Churn` (Target Variable: 'Yes' if churned, 'No' otherwise)
    * **Rows**: 7043
- **Features**: 21 columns (after dropping `customerID`)

### Target Variable
- `Churn`: 
  - `Yes`: customer has churned
  - `No`: customer is retained

---

## 3. Technologies Used

- **Language**: Python
- **Notebook**: Jupyter Notebook
- **Libraries**:
  - `pandas`, `numpy` – data manipulation
  - `matplotlib`, `seaborn` – data visualization
  - `scikit-learn` – preprocessing, ML models, evaluation
  - `xgboost` – advanced boosting models
  - `imblearn` – SMOTE for class imbalance handling
  - `pickle` – model serialization,saving encoders and models

---

## 📈 Project Workflow

### ✅ 1. Problem Definition
- Predict if a customer will churn.
- Churn means the customer is no longer subscribed to the service.

### ✅ 2. Data Loading
- CSV file `Telco-Customer-Churn.csv` was loaded using `pandas`.

### ✅ 3. Data Cleaning & Preprocessing
- `customerID` column dropped (non-predictive).
- `TotalCharges` column was originally of type `object`.
  - Invalid string entries (`" "`, `"<NA>"`, `""`, `"nan"`) were removed.
  - Column successfully converted to `float64`.
- No missing values remain after cleaning.

### ✅ 4. Target Distribution Analysis
- `Churn` column shows **class imbalance**:
  - Majority: `No` (non-churners)
  - Minority: `Yes` (churners)
- This imbalance will be handled using **SMOTE** in preprocessing.

### ✅ 5. Numerical Feature Analysis (Ongoing)
- Histograms plotted with KDE curves for features like:
  - `MonthlyCharges`
  - `TotalCharges`
  - `tenure`

---

## 📊 Sample Visualizations

> *(Add graphs/screenshots if uploading to GitHub)*

- `countplot()` for target distribution
- Histograms for numerical features

---

## 🧠 Insights So Far

1. `customerID` removed as it's not useful for modeling.
2. Handled invalid entries in `TotalCharges` and converted it to float.
3. No missing values remain in the dataset.
4. Target variable `Churn` is imbalanced and needs balancing.
5. Initial visual analysis of numerical features completed.

---

## 🛠️ Next Steps

- Encode categorical features using Label Encoding / One-Hot Encoding
- Balance the data using SMOTE
- Train multiple models (Decision Tree, Logistic Regression, XGBoost)
- Evaluate using F1-score, ROC AUC, confusion matrix
- (Optional) Deploy on Azure / Flask API

---
## 🔗 Resources

- Dataset Source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- YouTube Tutorial Inspiration: IR Siddharthan – Project #22



