# download_data.py
import os
import subprocess
from dotenv import load_dotenv


load_dotenv("cred.env")

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

DATASET = "rohanrao/formula-1-world-championship-1950-2020"
OUTPUT_DIR = "./data/raw"


os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Downloading {DATASET} into {OUTPUT_DIR}...")
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", DATASET,
    "-p", OUTPUT_DIR,
    "--unzip"
])

print("----- Download complete ------")