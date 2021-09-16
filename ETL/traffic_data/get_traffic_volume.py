import requests
import pandas as pd
import os
from itertools import product
import psycopg2
import numpy as np
import time
import datetime
from datetime import date


reg_points = pd.read_csv(r'C:\Users\Inger Lise\Documents\Luft og trafikk\files\registration_points.csv')

def get_traffic_data(start_date, end_date):
    data_list = []
    reg_list = reg_points['id'].tolist()
    sk_list = reg_points['sk_id'].tolist()
    for reg_point, sk_id in zip(reg_list, sk_list):
        url = 'https://www.vegvesen.no/trafikkdata/api/'
        headers =  {
            "content-type": "application/json",
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36',
            "Accept-Encoding": "*",
            "Connection": "keep-alive"}
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
        try:
            r = requests.post(url, json = {'query':query}, headers = headers)
            print(reg_point)
            if r.status_code == 200:
                json_data = r.json()
                for point in json_data['data']['trafficData']['volume']['byDay']['edges']:
                    if point != []:
                        point_dict = {}
                        
                        point_dict['reg_points'] = sk_id + 1
                        point_dict['date'] = point['node']['from']
                        
                        if point['node']['total']['volumeNumbers'] != None:
                            if len(point['node']['total']) > 1:
                                point_dict['volume'] = point['node']['total']['volumeNumbers']['volume']
                                point_dict['coverage'] = point['node']['total']['coverage']['percentage']
                            else:
                                point_dict['volume'] = point['node']['total']['volumeNumbers']['volume']
                                point_dict['coverage'] = None
                        else:
                            point_dict['volume'] = None
                            point_dict['coverage'] = 0
                        data_list.append(point_dict)
                        
        except Exception as e:
            msg = "Exception is:\n %s \n" % e
            print(msg)
            r = requests.post(url, json = {'query':query}, headers = headers)
            print(reg_point)
            if r.status_code == 200:
                json_data = r.json()
                for point in json_data['data']['trafficData']['volume']['byDay']['edges']:
                    if point != []:
                        point_dict = {}
                        
                        point_dict['reg_points'] = sk_id + 1
                        point_dict['date'] = point['node']['from']
                        
                        if point['node']['total']['volumeNumbers'] != None:
                            if len(point['node']['total']) > 1:
                                point_dict['volume'] = point['node']['total']['volumeNumbers']['volume']
                                point_dict['coverage'] = point['node']['total']['coverage']['percentage']
                            else:
                                point_dict['volume'] = point['node']['total']['volumeNumbers']['volume']
                                point_dict['coverage'] = None
                        else:
                            point_dict['volume'] = None
                            point_dict['coverage'] = 0
                        data_list.append(point_dict)
                        
    df = pd.DataFrame(data_list)
    df['date'] = pd.to_datetime(df['date'], errors = 'coerce', utc = True)
    df['date'] = df['date'].dt.strftime('%Y%m%d').astype(int)
    return df

start_date = pd.date_range('2017-01-01T12:00:00', periods = 18, freq = pd.offsets.MonthBegin(3), tz = 'Europe/Oslo')
end_date = pd.date_range('2017-03-31T12:00:00', periods = 18, freq = pd.offsets.MonthEnd(3), tz = 'Europe/Oslo')
weeks = pd.DataFrame({'from':start_date, 'to':end_date})
weeks['from'] = pd.to_datetime(weeks['from'])
weeks['to'] = pd.to_datetime(weeks['to'])
weeks['from'] = weeks["from"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
weeks['to'] = weeks["to"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

df = get_traffic_data('2021-06-30T12:00:00+00:00', '2021-09-07T12:00:00+00:00')

df_template = df.copy()
for index, row in weeks.iterrows():
    print('Waiting...')
    time.sleep(10)
    print('Cycle ' + str(index))
    temp_df = get_traffic_data(row['from'], row['to'])
    print('Writing...')
    temp_df.to_csv(r'files\traffic_data.csv', mode = 'a', index = False, header = not os.path.exists('files'))
    print('Concating...')
    df_template = pd.concat([df_template, temp_df])
    
df_postgres = df_template.copy()
df_postgres = df_postgres.reset_index(drop = True)


    
    
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

def update_facts_table_traffic():
    to_date = (date.today()-datetime.timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S+00:00')
    connection = get_connection()
    
    with connection.cursor() as traffic_cursor:
        traffic_cursor.execute('select max(sk_date) from facts_traffic')
        last_date_tuple = traffic_cursor.fetchone()
        for last_date in last_date_tuple:
            from_date = pd.to_datetime(str(last_date),format='%Y%m%d')
        from_date_plus = (datetime.datetime.strptime(str(from_date), '%Y-%m-%d %H:%M:%S')+datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S+00:00')
    
    df_update = get_traffic_data(from_date_plus, to_date)
    
    with connection.cursor() as traffic_cursor:
        for index, row in df_update.iterrows():
            traffic_cursor.execute(
        """
        insert into facts_traffic (sk_date, sk_traffic_reg, volume, coverage)
        values (%s, %s, %s, %s)
        """, (row['date'], row['reg_points'], row['volume'], row['coverage']))
        connection.commit()

update_facts_table_traffic()
