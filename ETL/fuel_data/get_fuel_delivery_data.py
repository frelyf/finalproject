import requests
import pandas as pd
from pyjstat import pyjstat
import psycopg2
import numpy as np

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

del df_fuel_delivery['purchaser group']
del df_fuel_delivery['region']
del df_fuel_delivery['contents']

def date_converter(string):
    year, month = string.split('M')
    date = f'{year}{month}01'
    return date

df_fuel_delivery['month'] = df_fuel_delivery['month'].apply(date_converter)

df_fuel_delivery = pd.pivot_table(df_fuel_delivery, values = 'value', index = ['petroleum products', 'month']).reset_index()

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
    create table if not exists fuel_delivery (
        id serial primary key,
        petroleum_product text not null,
        month integer not null,
        fuel_1000000_liters integer not null)
    """)
    connection.commit()
    
with connection.cursor() as fuel_cursor:
    for index, row in df_fuel_delivery.iterrows():
        fuel_cursor.execute(
        """
        insert into fuel_delivery (petroleum_product, month, fuel_1000000_liters)
        values (%s, %s, %s)
        """, (row['petroleum products'], row['month'], row['value']))
    connection.commit()