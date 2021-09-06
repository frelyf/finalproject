from numpy.lib.twodim_base import diagflat
import requests
import pandas as pd

reg_points = pd.read_csv(r'C:\Users\Fredrik Lyford\Documents\GitHub\finalproject\registration_points.csv')



def get_traffic_data():
    data_list = []
    reg_list = reg_points['id'].tolist()
    for reg_point in reg_list:
        url = 'https://www.vegvesen.no/trafikkdata/api/'
        headers =  {"content-type": "application/json"}
        query = f'''query {{
        trafficData(trafficRegistrationPointId: "{reg_point}") {{
            volume {{
            byDay(
                from: "2021-08-01T12:00:00+02:00"
                to: "2021-08-08T12:00:00+02:00"
            ) {{
                edges {{
                node {{
                    from
                    to
                    total {{
                    volumeNumbers {{
                        volume
                    }}
                    coverage {{
                        percentage
                    }}
                    }}
                }}
                }}
            }}
            }}
        }}
        }}'''

        r = requests.post(url, json = {'query':query}, headers = headers)
        if r.status_code == 200:
            json_data = r.json()
            for point in json_data['data']['trafficData']['volume']['byDay']['edges']:
                if point['node']['total']['volumeNumbers'] != None:
                    point_dict = {}
                    point_dict['reg point'] = reg_point
                    point_dict['date'] = point['node']['from']
                    point_dict['volume'] = point['node']['total']['volumeNumbers']['volume']
                    point_dict['coverage'] = point['node']['total']['coverage']['percentage']
                    data_list.append(point_dict)

    return data_list

data = get_traffic_data()
df = pd.DataFrame(data)
df.to_csv('traffic_volume_august.csv', sep = ',')