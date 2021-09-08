import requests
import pandas as pd
import os
from itertools import product
import psycopg2
import numpy as np


reg_points = pd.read_csv(r'C:\Users\Fredrik Lyford\Documents\GitHub\finalproject\files\registration_points.csv')

def get_traffic_data(start_date, end_date):
    data_list = []
    reg_list = reg_points['id'].tolist()
    sk_list = reg_points['sk_id'].tolist()
    for reg_point, sk_id in zip(reg_list, sk_list):
        url = 'https://www.vegvesen.no/trafikkdata/api/'
        headers =  {"content-type": "application/json"}
        query = f'''query {{
        trafficData(trafficRegistrationPointId: "{reg_point}") {{
            volume {{
            byDay(
                from: "{start_date}"
                to: "{end_date}"
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
        print(r.status_code)
        if r.status_code == 200:
            json_data = r.json()
            for point in json_data['data']['trafficData']['volume']['byDay']['edges']:
                if point != []:
                    point_dict = {}
                    
                    point_dict['reg_points'] = sk_id + 1
                    point_dict['date'] = point['node']['from']
                    
                    if point['node']['total']['volumeNumbers'] != None:
                        point_dict['volume'] = point['node']['total']['volumeNumbers']['volume']
                        point_dict['coverage'] = point['node']['total']['coverage']['percentage']
                    
                    else:
                        point_dict['volume'] = None
                        point_dict['coverage'] = 0
                    data_list.append(point_dict)
    
    df = pd.DataFrame(data_list)
    df['date'] = pd.to_datetime(df['date'], errors = 'coerce', utc = True)
    df['date'] = df['date'].dt.strftime('%Y%m%d').astype(int)
    return df

# start_date = pd.date_range('2017-01-01T12:00:00', periods = 24, freq = pd.offsets.MonthBegin(3), tz = 'Europe/Oslo')
# end_date = pd.date_range('2017-03-31T12:00:00', periods = 24, freq = pd.offsets.MonthEnd(3), tz = 'Europe/Oslo')
# weeks = pd.DataFrame({'from':start_date, 'to':end_date})
# weeks['from'] = pd.to_datetime(weeks['from'])
# weeks['to'] = pd.to_datetime(weeks['to'])
# weeks['from'] = weeks["from"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
# weeks['to'] = weeks["to"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

df = get_traffic_data('2017-01-01T12:00:00+00:00', '2021-06-30T12:00:00+00:00')

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

def nan_to_null(f,
        _NULL=psycopg2.extensions.AsIs('NULL'),
        _Float=psycopg2.extensions.Float):
    if not np.isnan(f):
        return _Float(f)
    return _NULL

psycopg2.extensions.register_adapter(float, nan_to_null)

with connection.cursor() as traffic_cursor:
    for index, row in df.iterrows():
        traffic_cursor.execute(
        """
        insert into facts_traffic (sk_date, sk_traffic_reg, volume, coverage)
        values (%s, %s, %s, %s)
        """, (row['date'], row['reg_points'], row['volume'], row['coverage']))
        print(index)
    connection.commit()


# reg_points = df['reg_points'].unique().tolist()
# dates = df['date'].unique().tolist()
# df_product = pd.DataFrame(list(product(reg_points, dates)), columns = ['reg_points','date'])
# df_product = df_product.merge(df, how='left', on = ['reg_points','date'])
# df_product['volume'].isna().sum()
# missing_values = df.isna().groupby(df['reg_points'], sort=False).sum()
# df.to_csv(r'files\traffic_data.csv', mode = 'a', index = False, header = not os.path.exists('files'))

