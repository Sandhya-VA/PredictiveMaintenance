# Predictive Maintenance for CNC Machines

Predicting machine failure using the AI4I 2020 Predictive Maintenance Dataset from UCI. The core challenge: failures are only 3.4% of records, so naive models get 96% accuracy by predicting "no failure" for everything — which is useless in practice.

**Best result:** 91% precision on the failure class using a stacking ensemble (Decision Tree + MLP + SVM → Logistic Regression meta-model), with SMOTE oversampling to handle class imbalance.

---

## The problem with imbalanced data

Without handling the imbalance, every model just predicts "no failure" and scores ~96% accuracy. That looks great on paper but catches zero actual failures. SMOTE generates synthetic failure samples during training so the model learns what failure actually looks like across the five failure modes in the dataset (TWF, HDF, PWF, OSF, RNF).

---

## Approach

```
Raw sensor data (ai4i2020.csv)
        |
Feature engineering + StandardScaler + OneHotEncoder
        |
SMOTE oversampling (3.4% → 50% failure rate in training)
        |
Train 3 base models:
  - Decision Tree
  - MLP Neural Network  
  - SVM (RBF kernel)
        |
5-fold stacking → Logistic Regression meta-model
        |
MLflow experiment tracking (precision, recall, F1 per run)
```

---

## Results

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Decision Tree | ~0.78 | ~0.72 | ~0.75 |
| MLP Neural Network | ~0.85 | ~0.79 | ~0.82 |
| SVM | ~0.88 | ~0.81 | ~0.84 |
| **Stacking Ensemble** | **~0.91** | **~0.83** | **~0.87** |

---

## Setup

```bash
pip install scikit-learn imbalanced-learn mlflow pandas numpy jupyter
```

Download the dataset from UCI:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Place `ai4i2020.csv` in the same directory as the notebook, then:

```bash
jupyter notebook "Predictive Maintenance.ipynb"
```

View MLflow experiment runs:
```bash
mlflow ui
# http://localhost:5000
```

---

## Files

- `Predictive Maintenance.ipynb` — full pipeline: EDA → preprocessing → SMOTE → model training → stacking → MLflow tracking
- `Predictive_Maintenance.pdf` — project report with methodology and analysis

---

`machine-learning` `python` `scikit-learn` `mlflow` `imbalanced-data` `smote` `ensemble` `predictive-maintenance` `mlops`
