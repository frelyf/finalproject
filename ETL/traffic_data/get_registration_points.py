import requests
import pandas as pd

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
df.to_csv('registration_points.csv', sep = ',')