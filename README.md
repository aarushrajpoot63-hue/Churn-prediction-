# Customer Churn Prediction using Machine Learning

**AIA Project – BCA (in collaboration with IBM), United University, Session 2026-27**
Submitted by: Aarush Singh Rajpoot (UU252010129) and Advay Mishra

## Problem statement
Predict whether a telecom customer will **churn (leave)** or **stay**, using customer details such as contract type, tenure, internet service and monthly charges. This is a supervised **binary classification** problem.

## Dataset
`data/telco_customer_churn.csv` – 7,006 rows × 14 columns (Telco customer-churn style).

> **Note on the data:** this file was *generated* with `src/make_dataset.py` to follow the structure of the popular Kaggle "Telco Customer Churn" dataset (same type of columns, missing values and duplicate rows). It is **not** the original Kaggle file. To use the real data, download it from Kaggle, save it as `data/telco_customer_churn.csv` (keep the column names) and re-run the notebook. Results will change.

## Project steps
1. Select the dataset
2. Choose the best model (Logistic Regression, Random Forest, Gradient Boosting compared with 5-fold CV)
3. Data cleaning and preprocessing – duplicates, missing values (SimpleImputer), Label Encoding, One-Hot Encoding, correlation analysis
4. Train / test split (80 / 20, stratified)
5. Model training + hyperparameter tuning (GridSearchCV)
6. Evaluation – accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, feature importance

## Results (test set, 1,400 customers)
| Metric | Value |
|---|---|
| Accuracy | 77.3% |
| Precision | 0.673 |
| Recall | 0.558 |
| F1-score | 0.610 |
| ROC-AUC | 0.834 |

Best model by cross-validation: **Logistic Regression** (tuned, C = 10). It won because the synthetic data was generated with a logistic relationship; on real Kaggle data, Random Forest or Gradient Boosting may do better. Top churn drivers: tenure, month-to-month contract, two-year contract.

## Folder structure
```
churn-prediction-project/
├── data/telco_customer_churn.csv
├── notebook/customer_churn_prediction.ipynb   # full project with outputs
├── src/make_dataset.py                        # creates the dataset
├── src/churn_prediction.py                    # script version of the notebook
├── images/                                    # saved charts
├── models/best_churn_model.pkl                # trained model
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
cd notebook
jupyter notebook customer_churn_prediction.ipynb
```

## Upload to GitHub
```bash
git init
git add .
git commit -m "Customer churn prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/churn-prediction-project.git
git push -u origin main
```
