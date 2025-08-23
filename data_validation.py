import os
import pandas as pd

# Path to your Kaggle dataset root
root_path = "./kaggle_datasets"

# List all folders
folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

# Get latest folder by last modified time
latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(root_path, f)))

print("📂 Latest Folder:", latest_folder)
print("📅 Timestamp:", os.path.getmtime(os.path.join(root_path, latest_folder)))

csv_path = os.path.join(root_path,latest_folder, 'customer_churn_dataset-training-master.csv')
print("📑 Reading file:", csv_path)

# Step 3: Load CSV into pandas
df = pd.read_csv(csv_path)

# -------------------------------
# 4. Run Validation Checks
# -------------------------------

print("\n🔎 Missing Values per Column:")
print(df.isnull().sum())

print("\n📊 Data Types:")
print(df.dtypes)

print("\n📌 Duplicate Rows:", df.duplicated().sum())

print("\n📈 Summary Statistics:")
print(df.describe())

# Negative check
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    negatives = (df[col] < 0).sum()
    if negatives > 0:
        print(f"⚠️ Column {col} has {negatives} negative values")

# Categorical values
cat_cols = df.select_dtypes(include=['object']).columns
print("\n📝 Unique Values in Categorical Columns:")
for col in cat_cols:
    print(f"{col}: {df[col].unique()}")

# Outlier detection (IQR)
print("\n🚨 Outlier Detection using IQR:")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

print("\n✅ Validation Complete!")

# -------------------------------
# Define Silver Layer Save Path
# -------------------------------
silver_root = ".\silver_layer"

# Keep same folder structure under silver_layer
silver_folder = os.path.join(silver_root, latest_folder)
os.makedirs(silver_folder, exist_ok=True)

# Save file with same name
silver_file = os.path.join(silver_folder, "customer_churn_dataset-training-master.csv")
df.to_csv(silver_file, index=False)

print(f"✅ DataFrame saved to: {silver_file}")
