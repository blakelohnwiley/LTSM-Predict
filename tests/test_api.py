import requests


method = "GETd"
base_url = "http://127.0.0.1:5000"
model_ver = str(1)
model_name = "BestModel_202213_1558"


if method == "GET":
  # METHOD GET
  url = f"{base_url}/train/BestModel"


  data = [
    {"Age":84,"Sex":"male","Embarked":"S"},
    {"Age":24,"Sex":"female","Embarked":"C"}
  ]
  payload = " [\n     {\"Age\":25,\"Sex\":\"male\",\"Embarked\":\"S\"},\n     {\"Age\":25,\"Sex\":\"male\",\"Embarked\":\"S\"}\n  ]"
  headers = {
    'Content-Type': 'application/json'
  }

  response = requests.request("GET", url, headers=headers, data=payload)

  print(response.text)

else:
  # METHOD POST

  endpoint = f"/predict/{model_ver}/{model_name}"
  data = [
    {"Age":84,"Sex":"male","Embarked":"S"},
    {"Age":24,"Sex":"female","Embarked":"C"}
  ]
  payload = " [\n     {\"Age\":25,\"Sex\":\"male\",\"Embarked\":\"S\"},\n     {\"Age\":25,\"Sex\":\"male\",\"Embarked\":\"S\"}\n  ]"
  headers = {
    'Content-Type': 'application/json'
  }

  response = requests.request("POST", f"{base_url}{endpoint}", headers=headers, data=payload)

  print(response.text)