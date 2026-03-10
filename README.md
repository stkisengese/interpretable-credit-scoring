# Credit Scoring — Interpretable Risk Prediction

## Overview
This project aims to develop a credit scoring model using the Home Credit dataset to predict the probability of loan default. The primary focus is on **interpretability**, ensuring that credit decisions are transparent and explainable to both regulators and customers.

## Key Features
- **Exploratory Data Analysis (EDA):** Deep dive into client data and historical credit records.
- **Predictive Modelling:** Gradient boosting models targeting an AUC-ROC ≥ 55% (goal ≥ 62%).
- **Explainable AI (XAI):** Global and local explanations using SHAP values.
- **Interactive Dashboards:** Visualising client risk profiles and feature contributions.

## Repository Structure
- `data/`: Contains the Home Credit datasets (CSV).
- `scripts/`: Core Python scripts for preprocessing, training, and prediction.
- `results/`: Model artifacts, reports, and client-level visualizations.
- `requirements.txt`: Project dependencies.
- `username.txt`: Project identifier.

## How to Run
1. **Setup Environment:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocessing:**
   ```bash
   python scripts/preprocess.py
   ```
3. **Training:**
   ```bash
   python scripts/train.py
   ```
4. **Scoring / Prediction:**
   ```bash
   python scripts/predict.py --client_id <SK_ID_CURR>
   ```

## Username
Project kaggle identifier: **skisenge01edukisumu**
