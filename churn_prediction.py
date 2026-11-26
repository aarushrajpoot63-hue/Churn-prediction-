"""Customer Churn Prediction - script version of the notebook."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# keep the csv in the data folder (or give the full path when running in Colab)
data = pd.read_csv("data/telco_customer_churn.csv")

data.head()

data.shape

# structural information: rows, columns, non-null values, data types, memory usage
data.info()

# statistical summary of numerical columns
data.describe()

# number of missing values in each column
data.isnull().sum()

# distribution of the target column (Churn)
print(data['Churn'].value_counts())
print()
print(data['Churn'].value_counts(normalize=True).round(3) * 100)
sns.countplot(x='Churn', data=data)
plt.title("Churn distribution")
plt.show()

df = data.copy()          # work on a copy so the original stays safe
print("Duplicate rows:", df.duplicated().sum())

df[df.duplicated()]

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print("Duplicate rows after removing:", df.duplicated().sum())
print("Shape:", df.shape)

from sklearn.impute import SimpleImputer

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_missing = ['InternetService', 'PaymentMethod']

df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
df[cat_missing] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_missing])

df.isnull().sum()

df.drop('customerID', axis=1, inplace=True)
df.columns

df.select_dtypes(include='object').nunique()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# check what Churn became: No -> 0, Yes -> 1
print(dict(zip(le.classes_, le.transform(le.classes_))))

# One-Hot Encoding for InternetService, Contract and PaymentMethod
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], dtype=int)
df.head()

print(df.shape)
df.dtypes

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, annot=False, linewidths=0.3)
plt.title("Correlation heatmap of all features")
plt.show()

# correlation of every feature with the target (Churn)
corr_target = df.corr()['Churn'].drop('Churn').sort_values()
print(corr_target.round(3))

plt.figure(figsize=(8, 7))
corr_target.plot(kind='barh', color=np.where(corr_target > 0, '#d9534f', '#337ab7'))
plt.title("Correlation of features with Churn")
plt.xlabel("Correlation")
plt.show()

# check highly correlated feature pairs (multicollinearity)
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
pairs = upper.stack().sort_values(ascending=False)
pairs[pairs > 0.7]

from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)    # input features
y = df['Churn']                 # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("y_train:", y_train.shape, "| y_test:", y_test.shape)
print("Churn ratio  train:", round(y_train.mean(), 3), " test:", round(y_test.mean(), 3))

from sklearn.linear_model import LogisticRegression
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
cv_table

best_name = cv_table.index[0]
print("Best model by cross-validation:", best_name)

from sklearn.model_selection import GridSearchCV

param_grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"max_depth": [5, 10, None], "min_samples_leaf": [1, 5, 10]},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
}

grid = GridSearchCV(models[best_name], param_grids[best_name], cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))

# final trained model
model = grid.best_estimator_
model

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
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
results

print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)']))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"{best_name} (AUC = {roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend()
plt.show()

# training vs testing score -> check for overfitting
print("Train accuracy:", round(model.score(X_train, y_train), 4))
print("Test accuracy :", round(model.score(X_test, y_test), 4))

from sklearn.inspection import permutation_importance

imp = permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=5, random_state=42)
importance = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
print(importance.head(10).round(4))

importance.head(10).sort_values().plot(kind='barh', color='#337ab7')
plt.title("Top 10 most important features")
plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
plt.show()

sample = X_test.iloc[[0]]
print("Predicted :", "Churn" if model.predict(sample)[0] == 1 else "Stay")
print("Churn probability:", round(model.predict_proba(sample)[0][1], 3))
print("Actual    :", "Churn" if y_test.iloc[0] == 1 else "Stay")

import joblib, os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_churn_model.pkl")
print("Model saved to models/best_churn_model.pkl")

print(f"Final model : {best_name} (tuned with GridSearchCV)")
print(f"Accuracy    : {results['Accuracy']:.2%}")
print(f"ROC-AUC     : {results['ROC-AUC']:.3f}")
print(f"Recall      : {results['Recall']:.2%}  -> share of churning customers we correctly catch")
print(f"Precision   : {results['Precision']:.2%}  -> share of predicted churners who really churn")
print("Top 3 churn drivers:", ", ".join(importance.index[:3]))
