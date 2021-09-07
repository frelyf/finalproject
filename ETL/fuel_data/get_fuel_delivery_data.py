import requests
import pandas as pd
from pyjstat import pyjstat

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

r = requests.post(url, json = query)
df_fuel_delivery = pyjstat.Dataset.read(r.text).write('dataframe')
