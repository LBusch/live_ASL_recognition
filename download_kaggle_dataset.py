import kagglehub
import os

# Requires Kaggle API token to be set up
# Check https://www.kaggle.com/docs/api for instructions

# === CONFIGURATION ===
# Specify the local directory where you want to download the dataset
DATA_DIR = os.path.join(os.getcwd())

# Dataset path from Kaggle
dataset_path = "grassknoted/asl-alphabet"

# Ensure the target directory exists
os.makedirs(DATA_DIR, exist_ok=True) 
    
# Temporarily set KAGGLEHUB_CACHE to DATA_DIR
prev_cache = os.environ.get("KAGGLEHUB_CACHE")
os.environ["KAGGLEHUB_CACHE"] = DATA_DIR

# check if the dataset is already downloaded
if os.path.exists(os.path.join(DATA_DIR, 'datasets', dataset_path.split('/')[-1])):
    print(f"Dataset {dataset_path} already exists in {DATA_DIR}.")

else:
    # Download the dataset
    local_dir = kagglehub.dataset_download(dataset_path, force_download=True)
    print(f"Dataset downloaded to: {local_dir}")

# Restore previous KAGGLEHUB_CACHE value
if prev_cache is not None:
    os.environ["KAGGLEHUB_CACHE"] = prev_cache
else:
    del os.environ["KAGGLEHUB_CACHE"]