# Loan Status Preprocessing & Balancing (TML-Project)

This repository contains the preprocessing pipeline for predicting loan defaults. The pipeline reads raw loan application data, cleans clutter, handles missing values, encodes categorical features, and balances the dataset using SMOTE (Synthetic Minority Over-sampling Technique) to prepare it for Machine Learning model ingestion.

---

## 🚀 Setup & Installation

Follow these steps to set up the project locally:

### Step 1: Download the Project from GitHub
Clone the repository or download the source code files to your local machine:
```bash
git clone <repository-github-url>
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
Run the following command to install the required libraries:
```bash
pip install pandas numpy scikit-learn imbalanced-learn jupyter
```

### Running the Notebook
Open the Jupyter notebook environment and run the pipeline:
```bash
jupyter notebook 1.ipynb
```
Execute each cell sequentially to preprocess, scale, and balance the data.

---

## 📋 Preprocessing Pipeline Overview

The notebook [1.ipynb](file:///c:/Users/User/Desktop/TML-Project/1.ipynb) performs the following tasks:

1. **Standardizing Headers:** Converts all column names to lowercase and replaces spaces with underscores.
2. **Target Variable Definition:** Flags loans as `0` (Good / Fully Paid) or `1` (Bad / Charged Off / Default), filtering out active loans.
3. **Data Leakage Removal:** Drops columns containing information generated *after* a loan is issued (e.g., `total_pymnt`, `recoveries`, `last_pymnt_amnt`).
4. **Data Cleansing & Imputation:** Drops identifiers, columns missing >50% data, and imputes remaining nulls (median for numbers, mode for categories).
5. **Categorical Encoding:** Performs target-specific numerical extraction, ordinal encoding for loan grades, and one-hot encoding for nominal categories.
6. **Train-Test Split & Scaling:** Splits data into 80% train and 20% test sets, scaling variables using `StandardScaler`.
7. **SMOTE Balancing:** Uses SMOTE on the training set to resolve class imbalance (equalizing Good vs. Bad loans).