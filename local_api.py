import json

import requests

# Send a GET using the URL http://127.0.0.1:8000
try:
    r = requests.get("http://127.0.0.1:8000")
    r.raise_for_status()

    # Print the status code
    print('Status code:', r.status_code)

    # Print the welcome message
    print("Welcome message: ", r.json()["greeting"])

except requests.exceptions.RequestException as e:
    print("Error occurred during request:", e)



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

try:
    # Send a POST using the data above
    r = requests.post("http://127.0.0.1:8000/data/", json=data)
    
    # Print the status code
    print("Status code:", r.status_code)
    
    # Print the result
    print("Result:", r.json()["result"])

except requests.exceptions.RequestException as e:
    print("Error occurred during request", e)
