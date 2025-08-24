import os
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

try:
    # 🔑 Step 1: Hardcode your credentials
    kaggle_username = "drajpant"   # <-- Replace with your username
    kaggle_key = "eae57c42a7b8e25cb0b51d2e01b9a220"   # <-- Replace with your API key
    dataset_slug = "muhammadshahidazeem/customer-churn-dataset"  # <-- Dataset slug
    base_dir = "kaggle_datasets"

    if not kaggle_username or not kaggle_key:
        raise ValueError("Kaggle username or API key is missing!")

    # 🔑 Step 2: Set environment variables
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    # 🔑 Step 3: Authenticate
    api = KaggleApi()
    api.authenticate()
    print("✅ Kaggle authentication successful.")

    # 🕒 Step 4: Create timestamped folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.abspath(base_dir)
    download_dir = os.path.join(base_dir, f"download_{timestamp}")
    os.makedirs(download_dir, exist_ok=True)

    # 📥 Step 5: Download dataset
    print(f"⬇️ Downloading dataset '{dataset_slug}' to {download_dir} ...")
    api.dataset_download_files(dataset_slug, path=download_dir, unzip=True)

    abs_path = os.path.abspath(download_dir)
    print(f"🎉 Dataset downloaded successfully to: {abs_path}")


except FileNotFoundError as e:
    print(f"❌ File/Directory error: {e}")
except ValueError as e:
    print(f"⚠️ Input Error: {e}")
except Exception as e:
    print(f"🚨 Unexpected error: {type(e).__name__} - {e}")