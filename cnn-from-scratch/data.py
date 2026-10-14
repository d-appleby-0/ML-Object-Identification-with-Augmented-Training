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

