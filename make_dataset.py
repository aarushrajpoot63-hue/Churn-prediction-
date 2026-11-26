"""Generates a Telco-style customer churn dataset (synthetic, Kaggle-style).
Replace data/telco_customer_churn.csv with the real Kaggle file if you wish --
the notebook works with the same column names."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 7000
gender = rng.choice(['Male','Female'], n)
senior = rng.choice([0,1], n, p=[.84,.16])
partner = rng.choice(['Yes','No'], n, p=[.48,.52])
dependents = rng.choice(['Yes','No'], n, p=[.3,.7])
tenure = np.clip(rng.gamma(2.2, 14, n).round(), 1, 72).astype(int)
phone = rng.choice(['Yes','No'], n, p=[.9,.1])
internet = rng.choice(['DSL','Fiber optic','No'], n, p=[.34,.44,.22])
contract = rng.choice(['Month-to-month','One year','Two year'], n, p=[.55,.21,.24])
paperless = rng.choice(['Yes','No'], n, p=[.59,.41])
payment = rng.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], n, p=[.34,.23,.22,.21])
base = np.where(internet=='Fiber optic', 70, np.where(internet=='DSL', 50, 22))
monthly = np.round(base + rng.normal(0, 8, n) + (phone=='Yes')*8, 2).clip(18, 120)
total = np.round(monthly*tenure*rng.uniform(.95,1.05,n), 2)
z = (-1.0 + 1.6*(contract=='Month-to-month') - 1.2*(contract=='Two year')
     + 0.8*(internet=='Fiber optic') - 0.045*tenure + 0.012*(monthly-65)
     + 0.5*(payment=='Electronic check') + 0.3*(senior==1) + 0.25*(paperless=='Yes')
     - 0.3*(dependents=='Yes') + rng.normal(0, .6, n))
churn = np.where(rng.random(n) < 1/(1+np.exp(-z)), 'Yes', 'No')
df = pd.DataFrame({'customerID':[f'CUST-{i:05d}' for i in range(n)],'gender':gender,'SeniorCitizen':senior,
 'Partner':partner,'Dependents':dependents,'tenure':tenure,'PhoneService':phone,'InternetService':internet,
 'Contract':contract,'PaperlessBilling':paperless,'PaymentMethod':payment,'MonthlyCharges':monthly,
 'TotalCharges':total,'Churn':churn})
df['tenure'] = df['tenure'].astype('float64')
for col,k in {'TotalCharges':35,'tenure':20,'InternetService':25,'MonthlyCharges':18,'PaymentMethod':15}.items():
    df.loc[rng.choice(n,k,replace=False), col] = np.nan
df = pd.concat([df, df.iloc[[3,30,300,1500,4000,6000]]], ignore_index=True)
df.to_csv('data/telco_customer_churn.csv', index=False)
print(df.shape, df.Churn.value_counts(normalize=True).round(3).to_dict())
