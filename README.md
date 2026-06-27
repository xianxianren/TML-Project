# Loan Status Preprocessing, Modeling & Explainability (TML-Project) 

This repository contains an end-to-end Machine Learning pipeline for predicting loan defaults. The pipeline processes raw LendingClub loan application data, resolves class imbalance using SMOTE, trains multiple classification models (Logistic Regression, Random Forest, XGBoost), integrates SHAP for Model Explainability (XAI), and optimizes model predictions using a custom business-oriented decision threshold to minimize financial defaults.

---

## 🚀 Setup & Installation

Follow these steps to set up the project locally:

### Step 1: Clone the Project
Clone the repository or download the source code files to your local machine:
```bash
git clone https://github.com/xianxianren/TML-Project
cd TML-Project
```
*(Alternatively, download the ZIP archive from GitHub and extract it to a directory of your choice).*

### Step 2: Download the Dataset
The dataset file (`loan.csv`) is too large (~441 MB) to be stored directly on GitHub.
* Download the dataset from the **[Project Instruction Link](https://drive.google.com/drive/folders/1ErYVe7mwGEMrXhwTqMF7K2HYxvPAODd1?usp=sharing)**.

### Step 3: Add the Dataset to the Project
Once the download is complete:
1. Locate the downloaded file (named `loan.csv`).
2. Copy and paste `loan.csv` directly into the root folder of this project (the folder containing `1.ipynb` and `README.md`).

---

## 🛠️ Requirements & Execution

### Prerequisites
Make sure you have Python 3.8+ and Jupyter installed.

### Install Dependencies
Run the following command to install the required libraries (including the machine learning models and interpretability tools):
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib jupyter
```

### Running the Notebook
Open the Jupyter notebook environment and run the pipeline:
```bash
jupyter notebook 1.ipynb
```
Execute each cell sequentially to run the preprocessing, training, explainability, and tuning steps.

---

## 📋 Machine Learning Pipeline Architecture

The notebook [1.ipynb](file:///c:/Users/User/Desktop/TML-Project/1.ipynb) is organized into three major phases:

### Phase 1: Preprocessing & Balancing (Cells 1–7)
1. **Standardizing Headers:** Converts all column names to lowercase and replaces spaces with underscores to prevent formatting syntax issues.
2. **Target Variable Definition:** Flags loans as `0` (Good / Fully Paid) or `1` (Bad / Charged Off / Default), filtering out active loans.
3. **Data Leakage Removal:** Drops features generated *after* a loan is issued (e.g., `total_pymnt`, `recoveries`, `last_pymnt_amnt`) to prevent artificial performance inflation.
4. **Data Cleansing & Imputation:** Drops unique identifiers and columns missing >50% data, then imputes remaining nulls (median for numerical values, mode for categorical values).
5. **Categorical Encoding:** Extracts numerical values from text columns (like `term` and `emp_length`), ordinally encodes loan grades (`grade`), and one-hot encodes nominal categoricals.
6. **Train-Test Split & Scaling:** Splits data into 80% train and 20% test sets, scaling variables using `StandardScaler` (learning parameters only from the training set).
7. **SMOTE Balancing:** Resolves the significant class imbalance (4:1 ratio of good vs. bad loans) by applying SMOTE only to the training set.

### Phase 2: Model Development & Evaluation (Cells 8–10)
We train and evaluate three baseline classification models on the balanced training data:
* **Logistic Regression:** Serves as a transparent linear baseline. Evaluates performance using standard metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and outputs the top 10 risk factors (regression coefficients).
* **Random Forest Classifier:** A non-linear bagging ensemble of decision trees. Captures feature interactions and outputs Gini-based feature importances.
* **XGBoost Classifier:** A gradient boosting model that trains sequentially to correct tree errors. Offers high predictive power but acts as a "black box" regarding feature impact direction.

### Phase 3: Explainable AI & SHAP Integration (Cell 11 & 13)
To achieve "Transparent Lending" and comply with regulatory requirements (such as explaining automated credit decisions), we integrate **SHAP (SHapley Additive exPlanations)**:
* **Global Explanation (Summary Plot):** Displays how each feature impacts the overall default risk score, including the positive/negative direction of impact.
* **Local Explanation (Waterfall Plot):** Explains exactly why a single specific applicant received their credit score breakdown, making automated decisioning auditable.

*Note: If initializing the SHAP Explainer with `model_output='probability'`, a background sample must be passed (e.g., `explainer = shap.TreeExplainer(model, data=X_train_sample, model_output='probability')`) to enable interventional feature perturbation.*

### Phase 4: Business Optimization & Tuning (Cell 12)
Standard ML models optimize for default accuracy (50% threshold), which can lead to catastrophic financial losses from false negatives (approving borrowers who default). We tune the XGBoost model for business impact:
1. **The Banker's Penalty (`scale_pos_weight`):** Penalizes missing a bad loan more heavily based on the ratio of good-to-bad loans.
2. **Custom Decision Threshold (30%):** Instead of the default 50% cutoff, we flag loans as "High Risk" if their predicted probability of default is **>= 30%**. This significantly boosts **Recall** (catching defaults before they happen) and protects the bank's capital.