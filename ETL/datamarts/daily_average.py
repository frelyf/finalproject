import psycopg2
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

query = """
    with td as (
	select 
		sk_date, 
		avg(volume) as traffic_volume
	from facts_traffic ft 
	group by sk_date
	order by sk_date),
wd as(
	select 
		sk_date, 
		avg(NULLIF("sum(precipitation_amount P1D)", 'NaN')) as precipitation, 
		avg(NULLIF("mean(wind_speed P1D)", 'NaN')) as wind_speed,
		avg(NULLIF("mean(air_temperature P1D)", 'NaN')) as air_temperature
	from facts_weather fw 
	group by sk_date 
	order by sk_date),
aqa as(
	select 
		sk_date,
		avg(NULLIF(pm2_5, 'NaN')) as pm2_5,
		avg(NULLIF(pm10, 'NaN')) as pm10,
		avg(NULLIF(nox, 'NaN')) as nox,
		avg(NULLIF(no2, 'NaN')) as no2,
		avg(NULLIF("no", 'NaN')) as "no"
	from facts_air_quality faq 
	group by sk_date 
	order by sk_date desc)
select 
	dd.dateid_serial,
   dd.month_actual,
   dd.year_actual,
	td.traffic_volume, 
	wd.precipitation,
	wd.wind_speed,
	wd.air_temperature,
	aqa.pm2_5,
	aqa.pm10,
	aqa.nox,
	aqa.no2,
	aqa."no"
from dim_date dd
join wd on wd.sk_date = dd.dateid_serial
join td on td.sk_date = dd.dateid_serial
join aqa on aqa.sk_date = dd.dateid_serial
order by dd.dateid_serial desc;
"""

# Build datamart in a dataframe df
def get_connection():
    connection = psycopg2.connect(
                host="trafikkluft.postgres.database.azure.com",                
                port="5432",
                user="postgres@trafikkluft",                
                password="Awesome1337",                
                database="postgres",            
            )
    return connection

engine = get_connection()
df = pd.read_sql_query(query, con = engine)

# Impute particulate matter based on monthly averages nearest neightbor
df_monthly_avg = df.copy()
df_monthly_avg = df_monthly_avg.groupby(['year_actual', 'month_actual']).mean()
df_monthly_avg = df_monthly_avg.reset_index()
imputer_month = KNNImputer(n_neighbors = 5)
imputer_month.fit(df_monthly_avg[['pm2_5','pm10']])
df_monthly_avg[['pm2_5','pm10']] = imputer_month.transform(df_monthly_avg[['pm2_5','pm10']])

df_monthly_avg['pm2_5_imputed'] = df_monthly_avg['pm2_5']
df_monthly_avg['pm10_imputed'] = df_monthly_avg['pm10']
df_monthly_avg = df_monthly_avg.drop(columns = ['dateid_serial', 'traffic_volume', 'precipitation', 'wind_speed', 'air_temperature', 'pm2_5', 'pm10', 'nox', 'no2', 'no'])

df_imputed = df.copy()
df_imputed = df_imputed.merge(df_monthly_avg, on = ['year_actual','month_actual'], how = 'left')
df_imputed['pm2_5'] = df_imputed['pm2_5'].fillna(df_imputed['pm2_5_imputed'])
df_imputed['pm10'] = df_imputed['pm10'].fillna(df_imputed['pm10_imputed'])
df_imputed = df_imputed.drop(columns = ['pm2_5_imputed','pm10_imputed'])

# Drop gas values
df_imputed = df_imputed.drop(columns = ['no','no2','nox'])

# Drop month and year
df_imputed = df_imputed.drop(columns = ['month_actual','year_actual'])
