import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "Pclass": 3,
    "Sex": 0,
    "Age": 25,
    "Fare": 7.25,
    "Embarked": 1
}

response = requests.post(url, json=data)
print("Статус:", response.status_code)
print("Ответ:", response.json())