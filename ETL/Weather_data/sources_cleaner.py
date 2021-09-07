# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 09:53:35 2021

@author: Abder
"""

import json
import csv

path = r"C:\Users\Abder\Desktop\json graduation\sources.json"
path2 = r"C:\Users\Abder\Desktop\json graduation\sources_clean.csv"


sources_list = []
with open(path) as crude_sources:
    data = json.load(crude_sources)
    
    for i in range(0,40):
        sources_list.append(data['data'][i]['id'])
        print(data['data'][i]['id'])
        
        

with open(file=path2, mode='w', newline='') as clean_sources:
    writer = csv.writer(clean_sources)
    for i in sources_list:
        writer.writerow([i])
    