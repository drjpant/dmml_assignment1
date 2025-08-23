import os
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
print('hellp')
# 🔁 STEP 1: Hardcode your credentials
kaggle_username = 'drajpant'  # <-- Replace with your username
kaggle_key = 'eae57c42a7b8e25cb0b51d2e01b9a220'        # <-- Replace with your API key
print('i m here')
# 🔁 STEP 2: Set environment variables programmatically
os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

# 🔁 STEP 3: Download dataset
api = KaggleApi()
api.authenticate()

dataset_slug = 'muhammadshahidazeem/customer-churn-dataset'  # Replace with your dataset slug
download_dir = './kaggle_datasets'

# 🕒 Create timestamped folder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_dir = os.path.abspath("kaggle_datasets")
download_dir = os.path.join(base_dir, f"download_{timestamp}")

os.makedirs(download_dir, exist_ok=True)
api.dataset_download_files(dataset_slug, path=download_dir, unzip=True)
abs_path = os.path.abspath(download_dir)

print(f"Dataset downloaded to: {abs_path}")