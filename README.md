# 🏦 Loan Default Prediction — End-to-End ML Pipeline with Explainable AI

An end-to-end Machine Learning pipeline for predicting loan defaults using LendingClub data. The pipeline covers data preprocessing, class imbalance handling (SMOTE), training of three classification models (Logistic Regression, Random Forest, XGBoost), model explainability via SHAP, and business-oriented decision threshold tuning.

---

## 📁 Project Structure

```
TML-Project/
├── 1.ipynb        # Main Jupyter Notebook (full ML pipeline)
├── loan.csv       # Dataset (must be downloaded separately — ~441 MB)
└── README.md      # Project documentation
```

### Notebook Cell Map (`1.ipynb`)

| Cell | Phase | Description |
|------|-------|-------------|
| 1 | Setup | Clean and categorized import of libraries (pandas, scikit-learn, xgboost, shap, etc.) |
| 2 | Step 1 | Load dataset (`loan.csv`) |
| 3 | Step 1 | Standardize column names to snake_case |
| 4 | Step 1 | Define & filter target variable (`loan_status` mapped to 0/1) |
| 5 | Step 2 | Remove 13 known data leakage columns (e.g. recoveries, total payments) |
| 6 | Step 3 | Clean useless IDs, drop high-missing columns (>50%), and impute null values |
| 7 | Step 4 | Encode categorical features (Term/Emp Length numeric extract, Grade mapping, One-Hot encoding) |
| 8 | Step 5 | Stratified Train-test split (80/20), StandardScaler scaling, and training-only SMOTE |
| 9 | Model 1 | Logistic Regression baseline with 5-Fold Stratified CV, 30-Seed robustness check, & coefficients analysis |
| 10 | Model 2 | Random Forest baseline with 5-Fold Stratified CV, 30-Seed robustness check, & Gini importance |
| 11 | Model 3a | Base XGBoost classifier with 5-Fold Stratified CV, 30-Seed robustness check, & built-in feature importance |
| 12 | SHAP 3a | SHAP explainers (Global Summary Plot, Local Waterfall, and Individual Applicant #0 probability conversion) |
| 13 | Summary | All 3 Models Robustness Summary (prints a comparison table of 30-seed average metrics) |
| 14 | Model 3b | Tuned XGBoost with class weight penalty (`scale_pos_weight`), 5-Fold CV, 30-Seed check, & 30% decision threshold |
| 15 | SHAP 3b | SHAP Global Summary, Local Waterfall, and Tuned Individual Applicant #0 evaluation |

---

## 🚀 Getting Started

Choose **Option A** (local setup) or **Option B** (Google Colab, no installation needed).

---

### ⚙️ Option A — Run Locally

#### Step 1: Download the Project from GitHub

Go to the [GitHub repository](https://github.com/xianxianren/TML-Project) and download the project:

**Method 1 — Git (recommended):**
```bash
git clone https://github.com/xianxianren/TML-Project.git
cd TML-Project
```

**Method 2 — ZIP Download:**
1. Click the green **`<> Code`** button on GitHub.
2. Click **Download ZIP**.
3. Extract the ZIP to a folder of your choice.

---

#### Step 2: Install Python

> Skip this step if you already have Python 3.8 or higher installed.

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download **Python 3.11** (recommended for best compatibility).
3. Run the installer.
   - ✅ **Important:** Check the box **"Add Python to PATH"** during installation.
4. Verify the installation by opening a terminal and running:
```bash
python --version
```

---

#### Step 3: Install Jupyter Notebook

Open a terminal (Command Prompt / PowerShell on Windows, Terminal on Mac/Linux) and run:
```bash
pip install jupyter
```

Verify by running:
```bash
jupyter --version
```

---

#### Step 4: Install Project Dependencies

Navigate to your project folder and install all required libraries:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib
```

#### Step 5: Download the Dataset

The dataset (`loan.csv`) is ~441 MB and is too large to store on GitHub.

1. Download it from the **[Project Dataset Link](https://drive.google.com/drive/folders/1ErYVe7mwGEMrXhwTqMF7K2HYxvPAODd1?usp=sharing)**.
2. Place `loan.csv` directly inside the project root folder (same folder as `1.ipynb`).

Your folder should look like this:
```
TML-Project/
├── 1.ipynb
├── loan.csv       ← place it here
└── README.md
```

---

#### Step 6: Launch Jupyter Notebook

In your terminal, navigate to the project folder and run:
```bash
jupyter notebook
```

Your browser will open. Click on **`1.ipynb`** to open the notebook, then run all cells from top to bottom using **`Kernel → Restart & Run All`**.

---

### ☁️ Option B — Run on Google Colab (No Installation Required)

Google Colab runs entirely in your browser — no Python or Jupyter installation needed.

#### Step 1: Upload the Notebook to Colab

1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Click **File → Upload Notebook**.
3. Upload the `1.ipynb` file from your downloaded project folder.

#### Step 2: Upload the Dataset

Once the notebook is open in Colab:

1. In the left sidebar, click the 📁 **Files** icon.
2. Click the **Upload** button and upload `loan.csv`.

> **Note:** Colab sessions are temporary. You will need to re-upload `loan.csv` every time you start a new session.

#### Step 3: Install Dependencies in Colab

Add a new code cell **at the very top** of the notebook and run:
```python
!pip install imbalanced-learn xgboost shap
```

> `pandas`, `numpy`, `scikit-learn`, and `matplotlib` are pre-installed in Colab.

#### Step 4: Run the Notebook

Run all cells from top to bottom using **`Runtime → Run all`**.

---

## 📋 Machine Learning Pipeline — Detailed Breakdown

### Phase 1: Data Preprocessing (Cells 1–8)

| Step | What It Does | Why It Matters |
|------|-------------|----------------|
| **Load Data** | Reads `loan.csv` (887,379 loans × 74 features) | Raw LendingClub data requires extensive cleaning |
| **Standardize Headers** | Lowercases column names, replaces spaces with underscores | Prevents syntax errors in column references |
| **Target Definition** | Maps `loan_status` → `0` (Fully Paid / Good) or `1` (Charged Off / Default / Bad); filters out active loans | Ensures only completed loans are analyzed |
| **Leakage Removal** | Drops 13 post-loan columns (e.g., `total_pymnt`, `recoveries`, `last_pymnt_amnt`) | Prevents artificially high accuracy from data that wouldn't exist at loan application time |
| **Cleaning & Imputation** | Drops columns missing >50% of data; fills numeric nulls with median, categorical nulls with mode | Removes unreliable features; ensures no missing values reach the model |
| **Feature Encoding** | Extracts numbers from `term`/`emp_length`; ordinal-encodes `grade` (A=1 to G=7); one-hot encodes remaining categoricals | Converts all data to numeric format required by ML models |
| **Train-Test Split** | 80% train / 20% test, stratified by target class | Preserves class imbalance ratio in both splits |
| **StandardScaler** | Normalizes all features to mean=0, std=1 (fitted on training data only) | Improves Logistic Regression convergence; prevents features with large ranges from dominating |
| **SMOTE** | Synthetically oversamples minority class (bad loans) in training data only | Fixes the 4:1 class imbalance so models learn to detect defaults |

**Dataset statistics after preprocessing:**
- Original: 887,379 rows × 74 columns
- After filtering active loans: 256,939 rows
- Final features after encoding: 47 columns
- Training set (before SMOTE): 205,551 rows
- Training set (after SMOTE): 335,538 rows (balanced 50/50)
- Test set: 51,388 rows

---

### Phase 2: Model Training & Evaluation (Cells 9–11)

Three models are trained on the SMOTE-balanced data and evaluated on the original (unbalanced) test set.

#### Model 1: Logistic Regression (Baseline)
- **Purpose:** Transparent linear baseline
- **Explainability:** Outputs top 10 risk factor coefficients
- **Typical results:** High recall, lower precision — catches many defaults but also flags some good loans

#### Model 2: Random Forest
- **Purpose:** Non-linear ensemble baseline
- **Explainability:** Gini-based feature importances
- **Typical results:** High precision, lower recall — more conservative; misses more defaults

#### Model 3a: XGBoost (Default Threshold — "The Black Box")
- **Purpose:** Gradient boosting model for highest predictive power
- **Problem:** Acts as a black box — built-in `feature_importances_` only shows which features were used for splits, but not the direction or magnitude of their effect
- **Solution:** SHAP in the next phase

**Validation Protocol:**
To ensure generalizability, stability, and zero data leakage, **every single model stage** (Logistic Regression, Random Forest, Base XGBoost, and Tuned XGBoost) is evaluated using these two rigorous methods:

1. **5-Fold Stratified Cross-Validation (`scaler -> SMOTE -> model`, all inside each fold)**: Splits the data into 5 stratified folds. Scaling and SMOTE-oversampling are applied *strictly inside each fold* to ensure no information leaks from validation folds into training.
2. **30-Seed Robustness Check (`full split -> scale -> SMOTE -> train -> test`, repeated)**: The entire train-test split, scaling, balancing, training, and testing cycle is repeated 30 times across 30 different random seeds to verify that model performance is stable and not a fluke of a single split.

Below are the **Robustness Summary** results (averaged over 30 seeds, showing `Mean ± SD`):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 64.68% ± 0.19% | 29.26% ± 0.18% | **64.97% ± 0.47%** | 40.35% ± 0.24% | 70.21% ± 0.25% |
| **Random Forest** | 79.35% ± 0.14% | 38.75% ± 0.57% | 21.23% ± 0.34% | 27.43% ± 0.36% | 69.06% ± 0.21% |
| **XGBoost (base)** | **81.50% ± 0.08%** | **48.02% ± 1.33%** | 8.02% ± 0.34% | 13.74% ± 0.53% | **70.25% ± 0.23%** |

---

### Phase 3: Explainable AI with SHAP (Cells 12 & 14)

SHAP (SHapley Additive exPlanations) reveals *how* and *why* the model makes each prediction — not just *what* it predicts.

#### Chart 1: SHAP Global Summary Plot
- Shows the **top features driving default risk across all test customers**
- Each dot represents one applicant
- **Color:** Red = high feature value; Blue = low feature value
- **X-axis position:** Positive = pushes toward default; Negative = pushes toward good repayment
- Answers: *"What is the bank's overall lending policy?"*

#### Chart 2: SHAP Local Waterfall Plot (Applicant #0)
- Explains **why a single applicant** received their predicted risk score
- Each bar shows how much one feature pushed the prediction up or down
- Answers: *"Why was this specific applicant flagged as high/low risk?"*

> **Technical Note:** The SHAP explainer uses `feature_perturbation="tree_path_dependent"` and `model_output="raw"` to avoid a known compatibility issue between newer SHAP versions and XGBoost's categorical-split path. This means SHAP explains XGBoost's **raw log-odds output** (not probabilities). The optional code at the end converts log-odds to a probability using `scipy.special.expit`.

---

### Phase 4: Business Optimization — Tuned XGBoost (Cell 13)

The base XGBoost model optimizes for accuracy (default 50% decision threshold), which leads to catastrophic financial outcomes: missing bad loans (false negatives) costs far more than incorrectly rejecting good applicants (false positives).

#### Tuning Strategy

**1. The Banker's Penalty (`scale_pos_weight`):**
```python
ratio = count(good_loans) / count(bad_loans)   # ≈ 4.44 in this dataset
xgb_tuned = XGBClassifier(scale_pos_weight=ratio, ...)
```
This tells XGBoost to penalize missing a bad loan 4.44× more than missing a good loan — aligning the model's optimization objective with the bank's financial interests.

**2. Model Hyperparameters:**
- `max_depth=4` — Shallower trees reduce overfitting and improve generalization
- `learning_rate=0.1` — Slower, more careful learning
- `n_estimators=200` — More trees to compensate for slower learning rate

**3. Custom Decision Threshold (30%):**
```python
y_pred = (y_pred_proba >= 0.30).astype(int)
```
Instead of flagging loans as "Bad" only when predicted probability ≥ 50%, we flag them at ≥ 30%. This significantly increases **Recall** (catching more actual defaults) at the cost of some **Precision** (more false alarms).

**Impact (30-Seed Robustness Average comparison):**

| Metric | Base XGBoost | Tuned XGBoost (30% threshold) | Change |
|--------|:---:|:---:|:---:|
| **Recall (catching defaults)** | 8.02% ± 0.34% | 49.26% ± 0.48% | **✅ +41.24%** (catching 6x more defaults) |
| **ROC-AUC (discrimination)** | 70.25% ± 0.23% | 70.09% ± 0.24% | **→ stable** (-0.16%) |
| **Accuracy (overall correctness)** | 81.50% ± 0.08% | 72.19% ± 0.28% | **↓ -9.31%** (expected trade-off) |

---

## ⚠️ Important Notes on Reproducibility

SHAP results (the exact feature rankings and contribution values) may differ slightly between machines even with the same code and `random_state=42`. This is caused by:

1. **CPU hardware** — XGBoost trains in parallel (`n_jobs=-1`). Floating-point arithmetic is non-associative, so different CPUs may produce tiny rounding differences that cascade into different tree splits.
2. **SMOTE version** — Different `scikit-learn` versions may select slightly different K-nearest neighbors, generating different synthetic training samples.

---

## 📦 Full Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
matplotlib
scipy
```