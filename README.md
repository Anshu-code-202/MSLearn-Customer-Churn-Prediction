
# MSLearn-Customer-Churn-Prediction
Predicting customer churn using ML (Telco dataset) — MS Learn project.

This project is part of Microsoft Learn’s AI/ML module. It focuses on predicting customer churn using classical machine learning algorithms like Decision Trees and XGBoost, along with data preprocessing, SMOTE for imbalance handling, and business insight extraction.

## 1. Dataset

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

## 2. Technologies Used

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



###  3. Data Cleaning & Preprocessing
- `customerID` column dropped (non-predictive).
- `TotalCharges` column was originally of type `object`.
  - Invalid string entries (`" "`, `"<NA>"`, `""`, `"nan"`) were removed.
  - Column successfully converted to `float64`.
- No missing values remain after cleaning.

###  4. Target Distribution Analysis
- `Churn` column shows **class imbalance**:
  - Majority: `No` (non-churners)
  - Minority: `Yes` (churners)
- This imbalance will be handled using **SMOTE** in preprocessing.

###  5. Numerical Feature Analysis (Ongoing)
- Histograms plotted with KDE curves for features like:
  - `MonthlyCharges`
  - `TotalCharges`
  - `tenure`

---


##  Sample Visualizations

> *(Add graphs/screenshots if uploading to GitHub)*

- `countplot()` for target distribution
- Histograms for numerical features

---

##  Insights So Far

1. `customerID` removed as it's not useful for modeling.
2. Handled invalid entries in `TotalCharges` and converted it to float.
3. No missing values remain in the dataset.
4. Target variable `Churn` is imbalanced and needs balancing.
5. Initial visual analysis of numerical features completed.

---
### 6. Encoding Categorical Features
Label Encoding: Used for binary columns (gender, Partner, Dependents, etc.)

One-Hot Encoding: Used for multi-category columns (InternetService, Contract, PaymentMethod, etc.)


###  7. Handling Class Imbalance with SMOTE
Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance class distribution in training data:



###  8. Model Training
Trained multiple machine learning models using the balanced training set:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

---

###  9. Cross-Validation (5-Fold)
Performed 5-Fold Cross-Validation using F1-score as the performance metric:
Collected F1-score results for each model for comparison.

---

###  10. Evaluation (Next Step)
Models will be evaluated using:

Confusion Matrix

F1 Score

ROC AUC Score

Classification Report

ROC Curve & Precision-Recall Curve Visualizations

*Evaluated on test set using:*

Accuracy Score

Confusion Matrix

Classification Report

ROC AUC Score


----

###  11. Model Saving (Planned)
Will serialize:

Trained models

Encoders / transformers

Using pickle for deployment or future use.

---

###  12. Predict on New Customer Data

Loaded saved model and encoders

Created new customer input as dictionary

Converted to DataFrame

Applied encoding using saved encoders

Ensured numeric columns were converted to float

Predicted churn using the trained model

---
 Resources
Dataset Source: Telco Customer Churn on Kaggle

YouTube Tutorial Inspiration: IR Siddharthan – Project #22



---
##  Resources

- Dataset Source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- YouTube Tutorial Inspiration: IR Siddharthan – Project #22



