# Zelestra-SolarAi
# Solar PV Efficiency & Power Output Prediction

<img src="TeamLogo.jpg" alt="Dragon Tech Logo" width="220"/>

Machine learning models for solar PV efficiency and power output prediction — developed as part of the **Zelestra x AWS ML Ascend Challenge (2nd Edition)**.

---

## 👥 Team

**DRAGON TECH**
- Sartaj Singh Virdi (`svirdi_be23@thapar.edu`)
- Prabhpreet Singh (`psingh9_be23@thapar.edu`) 
- Gurkirat Singh (`gsingh9_be23@thapar.edu`)

---

## ⚠️ Dataset Notice

This project was built using operational solar power plant data provided **exclusively** for the Zelestra x AWS ML Ascend Challenge.  

> **The dataset is proprietary and *not* included in this repository.  
> Redistribution or public sharing of the dataset is prohibited under competition terms.**

---

## 🚀 Project Goals

![Zelestra x AWS ML Ascend Challenge](66de7136452811f0.jpg)

- Predict **solar PV power output** and **efficiency** with high accuracy.
- Attribute losses to temperature, irradiance variability, and tracker performance.
- Develop and compare two advanced ensemble models for solar time-series data.

---

## 💡 Model Overview

We designed **two models** for this challenge:

### 1️⃣ Primary Model: Ensemble Stacking  
- Combines Ridge, ElasticNet, Bayesian Ridge, KNN, Random Forest, Extra Trees, Gradient Boosting, and SVR as base learners.  
- Meta-learner: Ridge Regression (optimized via cross-validation).  
- Features: Advanced time-aware engineering + interaction terms + polynomials.  
- Evaluation: 5-fold cross-validation stacking + hold-out test set.

### 2️⃣ Alternate Model: XGBoost + ElasticNet Hybrid  
- XGBoost for core predictions.  
- ElasticNet for residual correction.  
- Optimized hyperparameters and feature selection voting.

---

## 📊 Results Summary

| Metric                         | Ensemble Stacking | XGBoost + ElasticNet |
|---------------------------------|------------------|---------------------|
| R² (Target vs Predicted)        | 0.9995           | 0.76                |
| Correlation (Actual vs Pred)    | 0.94             | 0.95                |
| NRMSE                           | ~15%             | 15.5%               |
| MAPE                            | 36%              | High (due to low values) |

*Note: High MAPE is typical for solar data during low irradiance periods.*

---

## 🔑 Highlights

- Intelligent KNN imputation
- Datetime feature extraction + cyclical encoding
- Interaction + polynomial features
- Outlier detection (IQR + capping/removal)
- Multi-method feature selection
- Robust scaling
- Time-aware validation

---

## 📂 Repo Structure

- `Phase-1/` — Baseline model and dataset exploration  
- `Phase-2/EDA/` — Advanced exploratory analysis  
- `Phase-2/Preprocessing/` — Data cleaning + feature engineering  
- `Phase-2/Model/` — Final model notebooks (submission + trials)  

---

## 📝 Usage Example

```python
import pandas as pd

# Replace with your file path
df = pd.read_csv('your_solar_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

print(df.shape)
print(df.head())
