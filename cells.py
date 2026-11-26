cells = []
def md(s): cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("# Customer Churn Prediction using Machine Learning (Telco Customer Churn Dataset)")
md("""**Problem statement:** A telecom company loses revenue when customers leave (churn). Using customer details such as contract type, tenure, internet service and monthly charges, build a machine learning model that predicts whether a customer will churn (`Yes`) or stay (`No`).

**Type of problem:** Supervised learning – Binary Classification

**Dataset:** `telco_customer_churn.csv` – 7,000+ customers, 14 columns (Telco-style churn data; see README for the note on the data source).

**Project steps**
1. Select the dataset
2. Choose the best model
3. Data cleaning and preprocessing (missing values, duplicates, encoding, correlation)
4. Data splitting
5. Model training
6. Model evaluation""")

md("## 1. Importing libraries and loading the dataset")
code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")""")
code("""# keep the csv in the data folder (or give the full path when running in Colab)
data = pd.read_csv("../data/telco_customer_churn.csv")""")
code("data.head()")
code("data.shape")
code("""# structural information: rows, columns, non-null values, data types, memory usage
data.info()""")
code("# statistical summary of numerical columns\ndata.describe()")
code("# number of missing values in each column\ndata.isnull().sum()")
code("""# distribution of the target column (Churn)
print(data['Churn'].value_counts())
print()
print(data['Churn'].value_counts(normalize=True).round(3) * 100)
sns.countplot(x='Churn', data=data)
plt.title("Churn distribution")
plt.show()""")

md("""## 2. Choosing the best model
**Why this dataset needs a classification model:** the target `Churn` has two classes (Yes / No), so we need a *binary classifier*. The data is **tabular**, has a **mix of numerical and categorical features**, a few missing values, and a **moderately imbalanced** target (about 1 in 3 customers churn).

**Candidate models**

| Model | Why it is a candidate |
|---|---|
| Logistic Regression | Simple, fast, easy to explain – used as the *baseline* |
| Random Forest | Handles non-linear relationships and feature interactions, robust to outliers, gives feature importance |
| Gradient Boosting | Builds trees one after another to correct earlier errors; usually the strongest on tabular data |

Deep learning is **not** chosen here because the dataset is small and tabular – tree-based models usually match or beat neural networks in this situation while being faster and easier to interpret.

The **final model is selected by 5-fold cross-validation (ROC-AUC)** on the training data in Step 5, so the choice is based on results and not on assumption.""")

md("## 3. Data cleaning and preprocessing")
md("### 3.1 Handling duplicate records")
code("""df = data.copy()          # work on a copy so the original stays safe
print("Duplicate rows:", df.duplicated().sum())""")
code("df[df.duplicated()]")
code("""df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print("Duplicate rows after removing:", df.duplicated().sum())
print("Shape:", df.shape)""")
md("""### 3.2 Handling missing values
- Numerical columns → filled with the **median** (not affected by outliers)
- Categorical columns → filled with the **mode** (most frequent value)""")
code("""from sklearn.impute import SimpleImputer

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_missing = ['InternetService', 'PaymentMethod']

df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
df[cat_missing] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_missing])""")
code("df.isnull().sum()")
md("### 3.3 Removing unnecessary columns\n`customerID` is only an identifier and has no predictive value, so it is dropped.")
code("df.drop('customerID', axis=1, inplace=True)\ndf.columns")
md("""### 3.4 Encoding categorical data
- **Label Encoding** for columns with only 2 categories
- **One-Hot Encoding** for columns with 3 or more categories""")
code("df.select_dtypes(include='object').nunique()")
code("""from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# check what Churn became: No -> 0, Yes -> 1
print(dict(zip(le.classes_, le.transform(le.classes_))))""")
code("""# One-Hot Encoding for InternetService, Contract and PaymentMethod
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], dtype=int)
df.head()""")
code("print(df.shape)\ndf.dtypes")
md("### 3.5 Correlation analysis")
code("""plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, annot=False, linewidths=0.3)
plt.title("Correlation heatmap of all features")
plt.show()""")
code("""# correlation of every feature with the target (Churn)
corr_target = df.corr()['Churn'].drop('Churn').sort_values()
print(corr_target.round(3))

plt.figure(figsize=(8, 7))
corr_target.plot(kind='barh', color=np.where(corr_target > 0, '#d9534f', '#337ab7'))
plt.title("Correlation of features with Churn")
plt.xlabel("Correlation")
plt.show()""")
code("""# check highly correlated feature pairs (multicollinearity)
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
pairs = upper.stack().sort_values(ascending=False)
pairs[pairs > 0.7]""")
md("""**Observations from correlation analysis**
- `tenure` has a strong *negative* relationship with churn – longer-staying customers rarely leave.
- Month-to-month contracts and fiber-optic internet are positively related to churn; two-year contracts are negatively related.
- `TotalCharges` is strongly correlated with `tenure` (it is roughly tenure × monthly charges). Tree-based models are not harmed by this, and logistic regression is regularised, so all features are kept.""")

md("## 4. Data splitting\nThe data is split into **80% training** and **20% testing**. `stratify=y` keeps the same churn ratio in both sets.")
code("""from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)    # input features
y = df['Churn']                 # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("y_train:", y_train.shape, "| y_test:", y_test.shape)
print("Churn ratio  train:", round(y_train.mean(), 3), " test:", round(y_test.mean(), 3))""")

md("## 5. Model training")
md("### 5.1 Compare the candidate models using 5-fold cross-validation")
code("""from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000))]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

cv_scores = {}
for name, model in models.items():
    cv_scores[name] = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

cv_table = pd.DataFrame({"CV ROC-AUC": cv_scores}).sort_values("CV ROC-AUC", ascending=False).round(4)
cv_table""")
code("""best_name = cv_table.index[0]
print("Best model by cross-validation:", best_name)""")
md("### 5.2 Hyperparameter tuning of the best model (GridSearchCV)")
code("""from sklearn.model_selection import GridSearchCV

param_grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"max_depth": [5, 10, None], "min_samples_leaf": [1, 5, 10]},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
}

grid = GridSearchCV(models[best_name], param_grids[best_name], cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))""")
code("""# final trained model
model = grid.best_estimator_
model""")

md("## 6. Model evaluation")
code("""from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix, roc_curve)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

results = pd.Series({
    "Accuracy":  accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall":    recall_score(y_test, y_pred),
    "F1-score":  f1_score(y_test, y_pred),
    "ROC-AUC":   roc_auc_score(y_test, y_prob),
}).round(4)
results""")
code("print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)']))")
code("""cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.show()""")
code("""fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"{best_name} (AUC = {roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend()
plt.show()""")
code("""# training vs testing score -> check for overfitting
print("Train accuracy:", round(model.score(X_train, y_train), 4))
print("Test accuracy :", round(model.score(X_test, y_test), 4))""")
md("### Feature importance – which factors make customers leave?")
code("""from sklearn.inspection import permutation_importance

imp = permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=5, random_state=42)
importance = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
print(importance.head(10).round(4))

importance.head(10).sort_values().plot(kind='barh', color='#337ab7')
plt.title("Top 10 most important features")
plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
plt.show()""")
md("### Predicting for a new customer")
code("""sample = X_test.iloc[[0]]
print("Predicted :", "Churn" if model.predict(sample)[0] == 1 else "Stay")
print("Churn probability:", round(model.predict_proba(sample)[0][1], 3))
print("Actual    :", "Churn" if y_test.iloc[0] == 1 else "Stay")""")
code("""import joblib, os
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/best_churn_model.pkl")
print("Model saved to models/best_churn_model.pkl")""")
md("## 7. Conclusion")
code("""print(f"Final model : {best_name} (tuned with GridSearchCV)")
print(f"Accuracy    : {results['Accuracy']:.2%}")
print(f"ROC-AUC     : {results['ROC-AUC']:.3f}")
print(f"Recall      : {results['Recall']:.2%}  -> share of churning customers we correctly catch")
print(f"Precision   : {results['Precision']:.2%}  -> share of predicted churners who really churn")
print("Top 3 churn drivers:", ", ".join(importance.index[:3]))""")
md("""- The dataset was cleaned (duplicates removed, missing values imputed, `customerID` dropped) and encoded using Label Encoding and One-Hot Encoding.
- Three models were compared with cross-validation, the best was tuned, and it was tested on 20% unseen data.
- Accuracy alone can mislead on imbalanced data, so **precision, recall, F1 and ROC-AUC** were also used.
- **Future work:** handle class imbalance (SMOTE / `class_weight`), try XGBoost, and test on the real Kaggle dataset.""")
