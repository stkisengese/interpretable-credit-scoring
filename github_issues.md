# GitHub Issues — Credit Scoring Project

Execution Order

1 → Data loading
2 → EDA
3 → Feature engineering
4 → Model training
5 → Model evaluation
6 → SHAP explanations
7 → Client visualizations

---

## Issue #1 — Project Setup & Repository Structure
**Labels:** `setup` `infrastructure`
**Priority:** 🔴 High
**Milestone:** Week 1 — Day 1

### Description
Initialise the repository with the required structure and configuration files as defined in the project specification.

### Tasks
- [x] Create repository with the following directory structure:
  ```
  project/
  ├── README.md
  ├── username.txt
  ├── requirements.txt
  ├── data/
  ├── results/
  │   ├── model/
  │   ├── feature_engineering/
  │   ├── clients_outputs/
  │   └── dashboard/  (optional)
  └── scripts/
      ├── train.py
      ├── predict.py
      └── preprocess.py
  ```
- [x] Create `username.txt` following the format `username01EDU location_MM_YYYY` — **this file must not be modified after day 1**
- [x] Create `README.md` introducing the project, how to run the code, and displaying the username
- [x] Create `requirements.txt` listing all required libraries
- [x] Download and place all datasets in `data/` directory:
  - `application_train.csv` / `application_test.csv`
  - `bureau.csv` / `bureau_balance.csv`
  - `previous_application.csv`
  - `POS_CASH_balance.csv`
  - `credit_card_balance.csv`
  - `installments_payments.csv`
  - `HomeCredit_columns_description.csv`
- [x] Set up Kaggle account with username `username01EDU location_MM_YYYY`
- [x] Push Kaggle username profile description to Git on day 1

### Acceptance Criteria
- Repository structure matches specification exactly
- `username.txt` timestamp corresponds to first day of the project
- All datasets are accessible and loadable

---

## Issue #2 — Exploratory Data Analysis (EDA)
**Labels:** `eda` `data-analysis`
**Priority:** 🔴 High
**Milestone:** Week 1

### Description
Produce a fully commented `EDA.ipynb` notebook exploring all data sources, identifying patterns, quality issues, and opportunities for feature engineering.

### Tasks

**Target Variable Analysis**
- [x] Compute class distribution of TARGET — quantify imbalance ratio (expected: ~8–10% default rate)
- [x] Document the precise definition: late payment > X days on first Y installments

**Application Table (`application_train.csv`)**
- [x] Distribution plots for key numeric features: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_BIRTH`, `DAYS_EMPLOYED`
- [x] Default rate by categorical variables: `NAME_CONTRACT_TYPE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`
- [x] Analyse the 3 `EXT_SOURCE` features — correlation with TARGET, pairwise correlation between sources
- [x] Investigate anomalous values in `DAYS_EMPLOYED` (known issue: some values are 365243 — employed since far in the future)
- [x] Analyse address mismatch flags (`REG_REGION_NOT_LIVE_REGION` etc.) vs. default rate
- [x] Analyse credit bureau enquiry velocity features (`AMT_REQ_CREDIT_BUREAU_*`) vs. default rate
- [x] Explore `DAYS_LAST_PHONE_CHANGE` vs. default rate (fraud signal)
- [x] Analyse social circle default features (`DEF_30_CNT_SOCIAL_CIRCLE`, `DEF_60_CNT_SOCIAL_CIRCLE`)
- [x] Document all missing value patterns — distinguish structural missingness from data quality gaps

**Bureau & Bureau Balance**
- [x] Distribution of number of previous bureau credits per applicant
- [x] `CREDIT_ACTIVE` status breakdown and default rate per status
- [x] `CREDIT_TYPE` breakdown — car, cash, credit card, mortgage etc.
- [x] `bureau_balance STATUS` field — distribution of months per DPD bucket (0, 1, 2, 3, 4, 5, C, X)
- [x] Analyse `AMT_CREDIT_MAX_OVERDUE` and `AMT_CREDIT_SUM_OVERDUE` distributions

**Previous Applications**
- [x] `NAME_CONTRACT_STATUS` breakdown (Approved / Refused / Cancelled / Unused offer) vs. default rate
- [x] `CODE_REJECT_REASON` analysis — which reasons predict future default?
- [x] Application-to-credit gap: `AMT_APPLICATION` vs. `AMT_CREDIT`
- [x] Time between previous applications (`DAYS_DECISION`)

**Installment Payments**
- [x] Compute payment delay: `DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT` — distribution analysis
- [x] Compute underpayment: `AMT_PAYMENT / AMT_INSTALMENT` — distribution and relation to TARGET
- [x] Proportion of missed payments (rows with no `DAYS_ENTRY_PAYMENT`)

**Credit Card Balance**
- [x] Monthly utilisation: `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL` — trend over time
- [x] ATM cash withdrawal ratio as stress indicator
- [x] Minimum-payment-only behaviour flag

**Building / Neighbourhood Features**
- [x] Correlation matrix across `_AVG`, `_MODE`, `_MEDI` variants — assess redundancy
- [x] Correlation with TARGET — assess predictive value vs. noise

### Acceptance Criteria
- Notebook is fully commented and runs end-to-end without errors
- Each table is covered with at least 3 meaningful visualisations
- Key findings are summarised in markdown cells
- Missing value counts and patterns are documented for every table

---

## Issue #3 — Data Preprocessing Pipeline
**Labels:** `preprocessing` `pipeline`
**Priority:** 🔴 High
**Milestone:** Week 1–2

### Description
Build a reproducible preprocessing pipeline (`preprocess.py`) that handles all data quality issues identified during EDA, ready to feed into feature engineering.

### Tasks

**Missing Value Treatment**
- [ ] Fix `DAYS_EMPLOYED` anomaly — replace 365243 values with NaN, then impute or flag
- [ ] For `EXT_SOURCE_1/2/3` — impute missing values with median or model-based imputation; flag missingness as a binary feature (absence of bureau record is informative)
- [ ] For high-missingness building columns (`_AVG/_MODE/_MEDI`) — assess whether to impute, drop, or encode missingness as a feature
- [ ] For `OWN_CAR_AGE` — missing means no car; impute with 0 or flag explicitly
- [ ] Document imputation strategy chosen for each feature group

**Encoding**
- [ ] Binary encode flag columns (already 0/1 — verify consistency)
- [ ] Label encode or one-hot encode categorical variables (`NAME_CONTRACT_TYPE`, `NAME_INCOME_TYPE`, `CODE_GENDER`, etc.)
- [ ] Handle `CODE_GENDER` — verify `XNA` category and decide treatment

**Outlier Treatment**
- [ ] Cap or log-transform `AMT_INCOME_TOTAL` (heavy right skew)
- [ ] Review and cap extreme values in bureau amount columns

**Time Variable Handling**
- [ ] Document and enforce sign convention: all `DAYS_*` are negative integers (days before application)
- [ ] Convert `DAYS_BIRTH` to age in years: `abs(DAYS_BIRTH) / 365`
- [ ] Convert `DAYS_EMPLOYED` to years employed (after fixing anomaly)

**Pipeline Design**
- [ ] Wrap preprocessing steps in a `sklearn.pipeline.Pipeline` or `ColumnTransformer` for reproducibility
- [ ] Ensure pipeline can be applied identically to both train and test sets
- [ ] Save fitted pipeline as a serialisable object alongside the model

### Acceptance Criteria
- Preprocessing pipeline runs without errors on both train and test sets
- No data leakage from test set into preprocessing steps
- All transformations are reproducible and documented

---

## Issue #4 — Feature Engineering
**Labels:** `feature-engineering` `modelling`
**Priority:** 🔴 High
**Milestone:** Week 2

### Description
Engineer high-signal predictive features by aggregating and combining information across all 7 data tables. This is the most impactful step for model performance.

### Tasks

**Application Table — Ratio Features**
- [ ] `CREDIT_INCOME_RATIO` = `AMT_CREDIT / AMT_INCOME_TOTAL`
- [ ] `ANNUITY_INCOME_RATIO` = `AMT_ANNUITY / AMT_INCOME_TOTAL`
- [ ] `CREDIT_TERM` = `AMT_CREDIT / AMT_ANNUITY` (loan term in months)
- [ ] `GOODS_CREDIT_RATIO` = `AMT_GOODS_PRICE / AMT_CREDIT`
- [ ] `AGE_YEARS` = `abs(DAYS_BIRTH) / 365`
- [ ] `YEARS_EMPLOYED` = `abs(DAYS_EMPLOYED) / 365` (after anomaly fix)
- [ ] `EMPLOYMENT_RATIO` = `DAYS_EMPLOYED / DAYS_BIRTH` (employment vs. age)
- [ ] `EXT_SOURCE_MEAN` = mean of the three external scores
- [ ] `EXT_SOURCE_MIN` = minimum of the three external scores
- [ ] `EXT_SOURCE_PROD` = product of the three external scores
- [ ] `ADDRESS_MISMATCH_SCORE` = sum of all 6 address mismatch flags
- [ ] `ENQUIRY_RECENT` = sum of `AMT_REQ_CREDIT_BUREAU_HOUR` through `AMT_REQ_CREDIT_BUREAU_MON`
- [ ] `SOCIAL_CIRCLE_DEFAULT_RATE_30` = `DEF_30_CNT_SOCIAL_CIRCLE / OBS_30_CNT_SOCIAL_CIRCLE`
- [ ] `DOCUMENT_COUNT` = sum of all `FLAG_DOCUMENT_*` columns

**Bureau Aggregations (per `SK_ID_CURR`)**
- [ ] Count of total bureau credits, active credits, closed credits
- [ ] Max `CREDIT_DAY_OVERDUE` — worst overdue at time of application
- [ ] Sum and max of `AMT_CREDIT_MAX_OVERDUE` and `AMT_CREDIT_SUM_OVERDUE`
- [ ] `BUREAU_DEBT_RATIO` = `AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM`
- [ ] Count of credit types (diversity of credit history)
- [ ] Days since most recent bureau credit (`max(DAYS_CREDIT)`)

**Bureau Balance Aggregations (per `SK_ID_CURR`)**
- [ ] Count of months with STATUS in `{1,2,3,4,5}` (any delinquency)
- [ ] Count of months with STATUS = 5 (most severe — written off / 120+ DPD)
- [ ] Max DPD bucket ever observed
- [ ] Most recent month's STATUS
- [ ] Proportion of months with any delinquency

**Installment Payment Features (per `SK_ID_CURR`)**
- [ ] `PAYMENT_DELAY_MEAN` = mean of `DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`
- [ ] `PAYMENT_DELAY_MAX` = max payment delay ever
- [ ] `PAYMENT_RATIO_MEAN` = mean of `AMT_PAYMENT / AMT_INSTALMENT`
- [ ] `PAYMENT_RATIO_MIN` = minimum payment ratio (worst single payment)
- [ ] `LATE_PAYMENT_COUNT` = number of installments paid late
- [ ] `MISSED_PAYMENT_COUNT` = number of installments with no payment recorded
- [ ] Recent payment delay trend: compute delay over last 12 months vs. historical

**Credit Card Balance Features (per `SK_ID_CURR`)**
- [ ] `CC_UTILISATION_MEAN` = mean monthly utilisation rate
- [ ] `CC_UTILISATION_MAX` = peak utilisation
- [ ] `CC_ATM_DRAW_RATIO` = `AMT_DRAWINGS_ATM_CURRENT / AMT_DRAWINGS_CURRENT` (cash stress)
- [ ] `CC_MIN_PAYMENT_RATIO` = `AMT_PAYMENT_CURRENT / AMT_INST_MIN_REGULARITY`
- [ ] Max and mean DPD from `SK_DPD`

**POS/Cash Balance Features (per `SK_ID_CURR`)**
- [ ] Count of months with `SK_DPD > 0`
- [ ] Max `SK_DPD` across all months
- [ ] Remaining instalment proportion: mean `CNT_INSTALMENT_FUTURE / CNT_INSTALMENT`

**Previous Application Features (per `SK_ID_CURR`)**
- [ ] Count of approved, refused, cancelled applications
- [ ] Approval rate: `approved / total`
- [ ] `AMT_CREDIT_APPLICATION_GAP` = mean of `AMT_CREDIT - AMT_APPLICATION` (lender haircut)
- [ ] Number of distinct product types applied for
- [ ] Days since most recent previous application

### Acceptance Criteria
- All features are computed correctly with no look-ahead bias
- Feature table joins back to `SK_ID_CURR` cleanly with no duplicates
- Final feature matrix is documented with feature descriptions
- Feature engineering code is modular and reusable in `preprocess.py`

---

## Issue #5 — Model Training & Validation
**Labels:** `modelling` `training`
**Priority:** 🔴 High
**Milestone:** Week 2–3

### Description
Train the credit scoring model, validate it properly, and ensure it meets the minimum AUC ≥ 55% threshold on the Kaggle test set (target: ≥ 62%).

### Tasks

**Baseline Model**
- [ ] Train a logistic regression baseline as interpretability reference point
- [ ] Document why accuracy is not the right metric for this imbalanced problem (use AUC-ROC instead)
- [ ] Record baseline AUC on 5-fold cross-validation

**Primary Model — Gradient Boosting**
- [ ] Train LightGBM or XGBoost as primary model (both handle missing values natively)
- [ ] Use stratified k-fold cross-validation (k=5) to preserve class distribution across folds
- [ ] Tune key hyperparameters: learning rate, max depth, num leaves, min child samples, subsample
- [ ] Use `scale_pos_weight` (XGBoost) or `is_unbalance=True` / `class_weight` (LightGBM) to handle class imbalance

**Overfitting Prevention**
- [ ] Apply L1 and/or L2 regularisation (`reg_alpha`, `reg_lambda`)
- [ ] Implement early stopping on validation AUC with patience of 50–100 rounds
- [ ] Plot learning curves: training AUC vs. validation AUC across boosting rounds
- [ ] Plot learning curves: training AUC vs. validation AUC across training set sizes

**Evaluation**
- [ ] Report cross-validation AUC mean ± std
- [ ] Plot ROC curve with AUC annotation
- [ ] Plot Precision-Recall curve (more informative for imbalanced classes)
- [ ] Compute and report confusion matrix at a chosen operating threshold
- [ ] Justify threshold choice (e.g., balancing precision/recall for business context)

**Kaggle Submission**
- [ ] Generate predictions on `application_test.csv` using the trained pipeline
- [ ] Format submission as required: `SK_ID_CURR`, `TARGET` (probability)
- [ ] Submit and record public leaderboard AUC score

### Acceptance Criteria
- Cross-validation AUC ≥ 55% (target ≥ 62%)
- Learning curves demonstrate no significant overfitting gap
- Kaggle submission is successfully uploaded and scored
- All evaluation metrics are documented in `model_report.txt`

---

## Issue #6 — Model Report (`model_report.txt`)
**Labels:** `documentation` `reporting`
**Priority:** 🟠 Medium
**Milestone:** Week 3

### Description
Write the required `model_report.txt` describing the full methodology, covering the three mandatory sections from the project spec.

### Tasks
- [ ] **Algorithm section** — describe chosen algorithm, why it was selected over alternatives (interpretability vs. performance trade-off), key hyperparameters and their values
- [ ] **Why not accuracy** — explain class imbalance, the base rate fallacy (a model predicting all 0s achieves ~90%+ accuracy), and why AUC-ROC is the correct metric for ranking applicants by risk
- [ ] **Limits and improvements** — address at minimum:
  - Data limitations (no access to labelled test set for final evaluation)
  - Potential data drift over time (model trained on historical data)
  - Missing protected attribute analysis (fairness / fair lending considerations)
  - Features that could improve performance: more granular product data, macroeconomic indicators, behavioural biometrics
  - Possible model improvements: stacking, neural network embeddings for categorical features

### Acceptance Criteria
- File saved as `results/model/model_report.txt`
- All three mandatory sections are addressed clearly
- File is written for a non-technical compliance audience as well as data scientists

---

## Issue #7 — Global Model Interpretability (Feature Importance)
**Labels:** `interpretability` `shap`
**Priority:** 🔴 High
**Milestone:** Week 3

### Description
Implement global interpretability analysis to satisfy the compliance team's requirements — answering "what are the key variables driving the model overall?"

### Tasks
- [ ] Extract and plot built-in feature importance from LightGBM/XGBoost (gain-based importance)
- [ ] Compute SHAP values across the full training set using `shap.TreeExplainer`
- [ ] Plot global SHAP summary plot (beeswarm) — shows both importance and directionality
- [ ] Plot SHAP bar plot — mean absolute SHAP values ranked by importance
- [ ] Plot SHAP dependence plots for the top 5 most important features — show how feature value affects SHAP value, revealing non-linear relationships
- [ ] Produce a written narrative of the top 10 most important features, explaining in plain language what each means and why it makes intuitive sense for credit risk
- [ ] Verify no obviously discriminatory or nonsensical features appear near the top (regulatory check)
- [ ] Save all global plots in `results/clients_outputs/` or a dedicated `results/interpretability/` folder

### Acceptance Criteria
- Global SHAP summary plot is clear, labelled, and exportable
- Written narrative accompanies the plots
- Analysis confirms model relies on legitimate financial signals

---

## Issue #8 — Local Interpretability & Client Scoring Output
**Labels:** `interpretability` `shap` `client-output`
**Priority:** 🔴 High
**Milestone:** Week 3

### Description
Implement the per-client scoring programme that takes a model and customer ID as input and returns the predicted default probability, SHAP force plot, and Plotly visualisations.

### Tasks

**Core Scoring Programme (`predict.py`)**
- [ ] Accept `SK_ID_CURR` as input
- [ ] Load trained model and preprocessing pipeline
- [ ] Run feature engineering for the specified client
- [ ] Return predicted probability of default (the score)
- [ ] Compute SHAP value for this individual using `shap.TreeExplainer`

**SHAP Force Plot**
- [ ] Generate `shap.force_plot` or `shap.waterfall_plot` for the client
- [ ] Ensure the plot clearly shows which features push the score up (towards default) vs. down (away from default)
- [ ] Export force plot as HTML or embedded in PDF

**Plotly Visualisations**
- [ ] Panel 1 — **Client Profile**: bar/radar chart showing client's key variables (`EXT_SOURCE_MEAN`, `ANNUITY_INCOME_RATIO`, `CREDIT_INCOME_RATIO`, `AGE_YEARS`, `YEARS_EMPLOYED`, `PAYMENT_DELAY_MAX`)
- [ ] Panel 2 — **Client vs. Population Comparison**: for each key variable, show where this client sits relative to the full population distribution (e.g. violin plot with client marker, or percentile gauge)
- [ ] Panel 3 — **Score Gauge**: visual display of the probability score (0–100%) with risk band colouring (green / amber / red)
- [ ] Panel 4 — **Top Positive and Negative Factors**: horizontal bar chart of top 5 SHAP values, colour-coded by direction

**Three Required Client Analyses**

- [ ] **Client 1 — Correct prediction (train set)**: select a client the model confidently and correctly classifies. Document why the prediction is correct with reference to their feature values and SHAP plot. Save as `client1_correct_train.pdf`

- [ ] **Client 2 — Wrong prediction (train set)**: select a client the model misclassifies. Investigate and document *why* the model got it wrong — are there unusual feature combinations? Missing data? Save as `client2_wrong_train.pdf`

- [ ] **Client 3 — Test set client**: select any client from `application_test.csv`. Run full scoring and visualisation. Save as `client_test.pdf`

### Acceptance Criteria
- `predict.py` runs end-to-end for any valid `SK_ID_CURR`
- All three client PDFs are present in `results/clients_outputs/`
- SHAP force plot and Plotly panels are rendered correctly in each output
- Client 2 analysis includes a written explanation of the model's error

---

## Issue #9 — Dashboard (Bonus)
**Labels:** `dashboard` `bonus` `dash`
**Priority:** 🟡 Low (Bonus)
**Milestone:** Week 4 (if time permits)

### Description
Implement an interactive Dash dashboard that allows a user to enter any customer ID and retrieve the full scoring output and visualisations.

### Tasks
- [ ] Build `dashboard.py` using Plotly Dash
- [ ] Input: text field accepting `SK_ID_CURR`
- [ ] On submit: call `predict.py` scoring logic and render results
- [ ] Display components:
  - [ ] Score gauge (probability of default)
  - [ ] SHAP force / waterfall plot
  - [ ] Client vs. population comparison charts
  - [ ] Key variable profile panel
  - [ ] Top contributing factors bar chart
- [ ] Add error handling for invalid or missing customer IDs
- [ ] Document how to run the dashboard in `README.md`
- [ ] Save as `results/dashboard/dashboard.py`

### Acceptance Criteria
- Dashboard launches locally with `python dashboard.py`
- All visualisations render correctly for at least 5 different customer IDs
- No crashes on invalid inputs

---

## Issue #10 — Final Review, Testing & Submission
**Labels:** `qa` `submission`
**Priority:** 🔴 High
**Milestone:** Final Week

### Description
End-to-end review, code quality checks, and final Kaggle submission before project deadline.

### Tasks

**Code Quality**
- [ ] Ensure all scripts are commented and runnable by a reviewer from scratch
- [ ] Confirm `requirements.txt` is complete and pinned to specific library versions
- [ ] Test full pipeline on a clean environment: `preprocess.py` → `train.py` → `predict.py`
- [ ] Ensure `EDA.ipynb` runs end-to-end without errors

**Repository Checklist**
- [ ] `README.md` — complete, includes run instructions and Kaggle username
- [ ] `username.txt` — present, not modified after day 1
- [ ] `requirements.txt` — complete
- [ ] `EDA.ipynb` — commented and runnable
- [ ] `results/model/my_own_model.pkl` — trained model saved
- [ ] `results/model/model_report.txt` — all three sections complete
- [ ] `results/feature_engineering/` — feature engineering artifacts present
- [ ] `results/clients_outputs/client1_correct_train.pdf` — present
- [ ] `results/clients_outputs/client2_wrong_train.pdf` — present
- [ ] `results/clients_outputs/client_test.pdf` — present
- [ ] `scripts/train.py`, `predict.py`, `preprocess.py` — present and working

**Final Kaggle Submission**
- [ ] Generate final predictions with best model
- [ ] Submit to Kaggle and record final AUC score
- [ ] Document Kaggle score in `README.md` and `model_report.txt`

### Acceptance Criteria
- All deliverable files are present in the correct locations
- Kaggle AUC ≥ 55% (target ≥ 62%)
- Any reviewer can clone the repo and reproduce results by following `README.md`
