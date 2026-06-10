import requests

url = "https://www.openml.org/api/v1/json/data/61"
response = requests.get(url)

print(response.status_code)
print(response.json())