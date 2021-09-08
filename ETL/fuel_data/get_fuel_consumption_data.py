import requests
import pandas as pd
from pyjstat import pyjstat
import psycopg2
import numpy as np

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

r = requests.post(url, json = query)
df_fuel_consumption = pyjstat.Dataset.read(r.text).write('dataframe')

del df_fuel_consumption['contents']
del df_fuel_consumption['region']

df_fuel_consumption = pd.pivot_table(df_fuel_consumption, values = 'value', index = ['petroleum products', 'year']).reset_index()

def get_connection():
    connection = psycopg2.connect(
                host="trafikkluft.postgres.database.azure.com",                
                port="5432",                
                user="postgres@trafikkluft",                
                password="Awesome1337",                
                database="postgres",            
            )
    return connection

connection = get_connection()

with connection.cursor() as fuel_cursor:
    fuel_cursor.execute(
    """
    create table if not exists fuel_consumption (
        id serial primary key,
        petroleum_product text not null,
        year int4 not null,
        fuel_1000_liters integer not null)
    """)
    connection.commit()
    
with connection.cursor() as fuel_cursor:
    for index, row in df_fuel_consumption.iterrows():
        fuel_cursor.execute(
        """
        insert into fuel_consumption (petroleum_product, year, fuel_1000_liters)
        values (%s, %s, %s)
        """, (row['petroleum products'], row['year'], row['value']))
    connection.commit()
    
