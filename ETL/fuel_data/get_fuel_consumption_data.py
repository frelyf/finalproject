import requests

url = 'https://data.ssb.no/api/v0/en/table/11185/'

query = {
  "query": [
    {
      "code": "Region",
      "selection": {
        "filter": "item",
        "values": [
          "03"
        ]
      }
    },
    {
      "code": "PetroleumProd",
      "selection": {
        "filter": "item",
        "values": [
          "01",
          "02a",
          "02b"
        ]
      }
    }
  ],
  "response": {
    "format": "json-stat2"
  }
}

r = requests.get(url, json = query).json()

print(r)