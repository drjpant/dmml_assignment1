import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)

try:
    # Path to your Kaggle dataset root
    root_path = "./gold_layer"

    # List all folders
    folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    # Get latest folder by last modified time
    latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(root_path, f)))

    print("📂 Latest Folder:", latest_folder)
    print("📅 Timestamp:", os.path.getmtime(os.path.join(root_path, latest_folder)))

    csv_path = os.path.join(root_path, latest_folder, 'customer_churn_dataset-training-master.csv')
    print("📑 Reading file:", csv_path)

    # Load dataset
    df = pd.read_csv(csv_path)

    # Prepare data
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Start an MLflow run
    with mlflow.start_run(run_name="logistic_regression_churn"):
        # Define model
        lr_model = LogisticRegression(random_state=42, max_iter=1000)

        # Train model
        lr_model.fit(X_train, y_train)

        # Predictions
        y_pred = lr_model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log model (optional if autolog not used)
        mlflow.sklearn.log_model(lr_model, artifact_path="model")

    print("✅ MLflow run completed!")

    # Print evaluation results
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred, average='binary'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='binary'):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='binary'):.2f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.2f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

except Exception as e:
    print("❌ Error occurred:", str(e))
