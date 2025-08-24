import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_score, recall_score, f1_score)

try:
    # -------------------------------
    # 1. Get Latest Silver Layer Folder
    # -------------------------------
    root_path = "./silver_layer"

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"❌ Root path not found: {root_path}")

    folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    if not folders:
        raise FileNotFoundError("❌ No folders found in silver_layer directory")

    latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(root_path, f)))
    print("📂 Latest Folder:", latest_folder)
    print("📅 Timestamp:", os.path.getmtime(os.path.join(root_path, latest_folder)))

    csv_path = os.path.join(root_path, latest_folder, 'customer_churn_dataset-training-master.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

    print("📑 Reading file:", csv_path)
    df = pd.read_csv(csv_path)

    # -------------------------------
    # 2. Data Inspection
    # -------------------------------
    print(df.head(5))
    print(df.tail(2))
    print(df.sample(2))
    print(df.info())
    print("Shape:", df.shape)
    print(df.describe().T)
    print("Missing Values:\n", df.isna().sum())
    print("Null %:\n", (df.isna().sum() / len(df)) * 100)

    # -------------------------------
    # 3. Handle Missing Values
    # -------------------------------
    df = df.dropna()  # Drop rows with missing values
    print("✅ Nulls after dropna:\n", df.isna().sum())

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("✅ Missing values handled")

    # -------------------------------
    # 4. Remove Duplicates & Drop ID
    # -------------------------------
    print("📌 Duplicate Rows Before:", df.duplicated().sum())
    df = df.drop_duplicates()
    if 'CustomerID' in df.columns:
        df = df.drop(['CustomerID'], axis=1)
    print("✅ Duplicates removed & CustomerID dropped")

    # -------------------------------
    # 5. Encoding
    # -------------------------------
    le = LabelEncoder()
    if 'Gender' in df.columns:
        df['Gender'] = le.fit_transform(df['Gender'])
    if 'Subscription Type' in df.columns:
        df['Subscription Type'] = le.fit_transform(df['Subscription Type'])

    if 'Contract Length' in df.columns:
        df = pd.get_dummies(df, columns=['Contract Length'], drop_first=True, dtype=int)

    print("✅ Encoding complete")

    # -------------------------------
    # 6. Scaling
    # -------------------------------
    numerical_cols = ['Gender', 'Subscription Type']
    if 'Contract Length_Monthly' in df.columns:
        numerical_cols.append('Contract Length_Monthly')
    if 'Contract Length_Quarterly' in df.columns:
        numerical_cols.append('Contract Length_Quarterly')

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print("✅ Scaling complete")

    # -------------------------------
    # 7. Feature Engineering
    # -------------------------------
    if 'Last Interaction' in df.columns:
        df['RecentlyActive'] = (df['Last Interaction'] < 5).astype(int)
    if 'Support Calls' in df.columns:
        df['HighSupportUser'] = (df['Support Calls'] > 5).astype(int)

    print("✅ Feature Engineering complete")

    # -------------------------------
    # 8. Save to Gold Layer
    # -------------------------------
    gold_root = "./gold_layer"
    gold_folder = os.path.join(gold_root, latest_folder)
    os.makedirs(gold_folder, exist_ok=True)

    gold_file = os.path.join(gold_folder, "customer_churn_dataset-training-master.csv")
    df.to_csv(gold_file, index=False)

    print(f"✅ DataFrame saved to: {gold_file}")

except Exception as e:
    print(f"❌ Error occurred: {e}")
