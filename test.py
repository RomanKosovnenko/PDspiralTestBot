import json
import requests

url = 'https://pddspytorch.azurewebsites.net/predict'


r = requests.get(url = url, json = {})
data = r.json()

reply.text = f"PyTorch predictor Status: {data['status']}"