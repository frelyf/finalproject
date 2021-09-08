
import requests
import csv
import pandas as pd
import psycopg2
import numpy as np

def nilf (search_terms):
    url = f'https://api.nilu.no/{search_terms}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return 'Nah'

#'lookup/areas'
#'lookup/stations'
#'aq/historical/2019-01-01/2020-01-01/Hjortnes' (from date, to date, station)
#'/obs/historical/{fromtime}/{totime}/{latitude}/{longitude}/{radius}'
    
oslo_stations_call = nilf('lookup/stations')

oslo_stations = []
for i in oslo_stations_call:
    if i['municipality'] == 'Oslo':
        oslo_stations.append(i)
        

oslo_stations_info = []
for i in oslo_stations:
    if i['lastMeasurment'] > '2017-01-01':
        oslo_stations_info_temp = {}
        oslo_stations_info_temp['id'] = i['id']
        oslo_stations_info_temp['type'] = i['type']
        oslo_stations_info_temp['station'] = i['station']
        oslo_stations_info_temp['lastMeasurement'] =  i['lastMeasurment']
        oslo_stations_info_temp['components'] = i['components']
        oslo_stations_info_temp['latitude'] = i['latitude']
        oslo_stations_info_temp['longitude'] = i['longitude']
        oslo_stations_info.append(oslo_stations_info_temp)
#print(oslo_stations_info)

oslo_station_names = []
for i in oslo_stations_info:
    oslo_station_names.append(i['station'])
#print(oslo_station_names)


def write_to_csv (search_terms):
    #data = nilf(search_terms)
    data = search_terms
    with open('oslo_stations_info.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data[0].keys())
        for i in data:
            writer.writerow(i.values())

write_to_csv(oslo_stations_info)
#write_to_csv('/obs/historical/2021-08-01/2021-08-07/Alnabru')

def obs_per_day (from_date, to_date, station_name):
    return nilf(f'/stats/day/{from_date}/{to_date}/{station_name}')

#obs_test = obs_per_day('2021-08-01','2021-08-07', 'Alnabru')
#print(obs_test)


def get_all_obs (from_date, to_date):
    all_obs_list = []
    for station_name in oslo_station_names:
        all_obs_list.append(nilf(f'/stats/day/{from_date}/{to_date}/{station_name}'))
    return all_obs_list

#all_obs_test = get_all_obs('2021-08-01','2021-08-07')
#print(all_obs_test)

def get_connection():
    connection = psycopg2.connect(
                host="trafikkluft.postgres.database.azure.com",                
                port="5432",                
                user="postgres@trafikkluft",                
                password="Awesome1337",                
                database="postgres",            
            )
    return connection

def python_table(create_table_query):
    conn = get_connection()
    with conn as connection:
        cursor = connection.cursor()
        cursor.execute(f'create table if not exists {create_table_query}')

#dim_air_quality
#python_table('dim_air_quality (sk_air_quality serial primary key, station_id int, station_type varchar, name varchar, lat float, lon float)')
#dim_traffic_reg
#python_table('dim_traffic_reg (sk_traffic_reg serial primary key, point_id varchar, name varchar, lat float, lon float)')
#facts_traffic
#python_table('facts_traffic (traffic_facts_id serial primary key, sk_date int references dim_date (dateid_serial), sk_traffic_reg int references dim_traffic_reg (sk_traffic_reg), volume int, coverage float, length_range varchar)')
#facts_air_quality
#python_table('facts_air_quality (air_quality_facts_id serial primary key, sk_date int references dim_date (dateid_serial), sk_air_quality int references dim_air_quality (sk_air_quality), PM2_5 float, PM10 float, NOx float, NO2 float, NO float)')

def insert_p_t_dim():
    data = oslo_stations_info
    conn = get_connection()
    with conn as connection:
        cursor = connection.cursor()
        for i in data:
            cursor.execute('insert into dim_air_quality (station_id, name, station_type, lat, lon) values (%s, %s, %s, %s, %s)', (i['id'], i['station'], i['type'], i['latitude'], i['longitude']))

insert_p_t_dim()
print(oslo_stations_info)

def insert_p_t_fact(to_date, from_date):
    data = get_all_obs(to_date, from_date)
    component_list = ['PM2.5', 'PM10', 'NOx', 'NO2', 'NO']
    station_dict = {'Alnabru':'1', 'Bryn skole':'2', 'Bygdøy Alle':'3','E6 Alna senter':'4', 'Grønland':'5','Hjortnes':'6', 'Kirkeveien':'7',
                    'Loallmenningen':'8', 'Manglerud':'9', '"Rv 4, Aker sykehus"':'10', 'Skøyen':'11', 'Smestad':'12', 
                    'Sofienbergparken':'13', 'Spikersuppa':'14', 'Vahl skole':'15', 'Åkebergveien':'16'}
    data_list = []
    for item in data:
        for element in item:
            if element['component'] in component_list:
                component = element['component']
                if element['station'] in station_dict.keys():
                    station_name = element['station']
                    id = station_dict.get(station_name)
                    for obs in element['values']:
                        data_dict = {}
                        date = ((obs['dateTime'])[0:10]).replace('-','')
                        data_dict['sk_date'] = date
                        data_dict['sk_air_quality'] = id
                        data_dict["component"] = component
                        data_dict['value'] = obs['value']
                        data_list.append(data_dict)
    df = pd.DataFrame(data_list)
    df = df.pivot_table(index = ['sk_date','sk_air_quality'], columns = ['component'], values = 'value').fillna(np.nan).reset_index()
    conn = get_connection()
    with conn as connection:
        cursor = connection.cursor()
        for index, row in df.iterrows():
            cursor.execute(f'insert into facts_air_quality (sk_date, sk_air_quality, NO, NO2, NOx, PM10, PM2_5) values (%s, %s, %s, %s, %s, %s, %s)', (row['sk_date'], row['sk_air_quality'], row['NO'], row['NO2'], row['NOx'], row['PM10'], row['PM2.5']))

#insert_p_t_fact('2015-01-01','2021-06-30')       

def get_time():
    conn = get_connection()
    with conn as connection:
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS dim_date(
    DATEID_SERIAL SERIAL PRIMARY KEY, 
	date_actual date not null,
    day_name varchar(9),
    day_of_week INT NOT NULL,
    day_of_month INT NOT NULL,
    day_of_quarter INT NOT NULL,
    day_of_year INT NOT NULL,
    week_of_month INT NOT NULL,
    week_of_year INT NOT NULL,
    week_of_year_iso CHAR(10) NOT NULL,
    month_actual INT NOT NULL,
    month_name VARCHAR(9) NOT NULL,
    month_name_abbreviated CHAR(3) NOT NULL,
    quarter_actual INT NOT NULL,
    quarter_name VARCHAR(9) NOT NULL,
    year_actual INT NOT NULL);''')

#get_time()

def insert_data_dim_date():
    conn = get_connection()
    with conn as connection:
        cursor = connection.cursor()
        cursor.execute('''insert into dim_date
            SELECT TO_CHAR(datum, 'yyyymmdd')::INT AS date_dim_id,
            datum AS date_actual,
            TO_CHAR(datum, 'Day') AS day_name,
            EXTRACT(ISODOW FROM datum) AS day_of_week,
            EXTRACT(DAY FROM datum) AS day_of_month,
            datum - DATE_TRUNC('quarter', datum)::DATE + 1 AS day_of_quarter,
            EXTRACT(DOY FROM datum) AS day_of_year,
            TO_CHAR(datum, 'W')::INT AS week_of_month,
            EXTRACT(WEEK FROM datum) AS week_of_year,
            EXTRACT(ISOYEAR FROM datum) || TO_CHAR(datum, '"-W"IW-') || EXTRACT(ISODOW FROM datum) AS week_of_year_iso,
            EXTRACT(MONTH FROM datum) AS month_actual,
            TO_CHAR(datum, 'Month') AS month_name,
            TO_CHAR(datum, 'Mon') AS month_name_abbreviated,
            EXTRACT(QUARTER FROM datum) AS quarter_actual,
            CASE
                WHEN EXTRACT(QUARTER FROM datum) = 1 THEN 'First'
                WHEN EXTRACT(QUARTER FROM datum) = 2 THEN 'Second'
                WHEN EXTRACT(QUARTER FROM datum) = 3 THEN 'Third'
                WHEN EXTRACT(QUARTER FROM datum) = 4 THEN 'Fourth'
                END AS quarter_name,
            EXTRACT(ISOYEAR FROM datum) AS year_actual
            FROM (SELECT '2015-01-01'::DATE + SEQUENCE.DAY AS datum
            FROM GENERATE_SERIES(0, 2372) AS SEQUENCE (DAY)
            GROUP BY SEQUENCE.DAY) DQ
            ORDER BY 1;''')
        connection.commit()

#insert_data_dim_date()