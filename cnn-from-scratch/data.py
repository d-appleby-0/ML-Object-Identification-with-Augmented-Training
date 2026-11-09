"""
import requests

API_KEY = Roboflow_API_key
workspace = Roboflow_workspace
project = Roboflow_project_name
version = 1
format = "yolov8"

url=f"https://api.roboflow.com/{workspace}/{project}/{version}/{format}"

response = requests.get(url, headers={"Authorization": f"Bearer {API_KEY}"})

if response.status_code == 200:
    with open("dataset.zip", "wb") as f:
        f.write(response.content)
    print("Dataset downloaded successfully.")
else:
    print(f"Error: {response.status_code} - {response.text}")
"""






# Temp refactor for testing:

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, seed=0):
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 1, 8, 8)            # (N, C, H, W)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def batches(X, y, batch_size, rng):
    perm = rng.permutation(len(X))
    for i in range(0, len(X), batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]

