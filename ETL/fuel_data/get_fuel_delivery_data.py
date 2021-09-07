import requests

url = 'https://data.ssb.no/api/v0/en/table/11174/'

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
      "code": "Kjopegrupper",
      "selection": {
        "filter": "item",
        "values": [
          "00"
        ]
      }
    },
    {
      "code": "PetroleumProd",
      "selection": {
        "filter": "item",
        "values": [
          "03",
          "04a",
          "04b"
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