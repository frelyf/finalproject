import requests
import pandas as pd
import psycopg2

url = 'https://www.vegvesen.no/trafikkdata/api/'
headers =  { "content-type": "application/json" }
query = '''	query {
	trafficRegistrationPoints(searchQuery: {countyNumbers: [3]}) {
    id
    name
    location {
      coordinates {
        latLon {
          lat
          lon
        }
      }
    }
  }
}
'''

json_data = requests.post(url, json = {'query':query}, headers = headers).json()

data_list = []

for point in json_data['data']['trafficRegistrationPoints']:
    point_dict = {}
    point_dict['id'] = point['id']
    point_dict['name'] = point['name']
    point_dict['lat'] = point['location']['coordinates']['latLon']['lat']
    point_dict['lon'] = point['location']['coordinates']['latLon']['lon']
    data_list.append(point_dict)

df = pd.DataFrame(data_list)

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

with connection.cursor() as traffic_cursor:
    for index, row in df.iterrows():
        traffic_cursor.execute(
        """
        insert into dim_traffic_reg (point_id, name, lat, lon)
        values (%s, %s, %s, %s)
        """, (row['id'], row['name'], row['lat'], row['lon']))
    connection.commit()