import os
import pandas as pd

# Path to your Kaggle dataset root
root_path = "./silver_layer"

# List all folders
folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

# Get latest folder by last modified time
latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(root_path, f)))

print("📂 Latest Folder:", latest_folder)
print("📅 Timestamp:", os.path.getmtime(os.path.join(root_path, latest_folder)))

csv_path = os.path.join(root_path,latest_folder, 'customer_churn_dataset-training-master.csv')
print("📑 Reading file:", csv_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve,precision_score, recall_score, f1_score



df = pd.read_csv(csv_path)
df

df.head(5)

df.tail()

df.sample()

df.info()

df.shape

df.describe().T

df.isna().sum()

# Calculate percentage of null values
(df.isna().sum() / len(df)) * 100

df = df.dropna()

df.isna().sum()

numerical_cols = df.select_dtypes(include=['int64', 'float64'])
print(f'numerical_cols: {numerical_cols}')



for col in numerical_cols:
    print(col)
    df[col] = df[col].fillna(df[col].mean())

    df[col] = df[col].fillna(df[col].mode()[0])   # for categories

    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object'])
print(f'object_cols: {numerical_cols}')

for col in categorical_cols:
    print(col)
    df[col] = df[col].fillna(df[col].mode()[0])
df.info()
df.duplicated().sum()

df = df.drop(['CustomerID'], axis=1)

df.head()


le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

df['Subscription Type'] = le.fit_transform(df['Subscription Type'])
df.info()
df.head()

df = pd.get_dummies(df, columns=['Contract Length'], drop_first=True, dtype=int)

df.head()

df.info()


numerical_cols=['Gender','Subscription Type','Contract Length_Monthly','Contract Length_Quarterly']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])





scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

df.info()

df.describe().T

df['RecentlyActive'] = (df['Last Interaction'] < 5).astype(int)

df['HighSupportUser'] = (df['Support Calls'] > 5).astype(int)

df.head()

# -------------------------------
# Define Silver Layer Save Path
# -------------------------------
gold_root = ".\gold_layer"

# Keep same folder structure under silver_layer
gold_folder = os.path.join(gold_root, latest_folder)
os.makedirs(gold_folder, exist_ok=True)

# Save file with same name
gold_file = os.path.join(gold_folder, "customer_churn_dataset-training-master.csv")
df.to_csv(gold_file, index=False)

print(f"✅ DataFrame saved to: {gold_file}")


