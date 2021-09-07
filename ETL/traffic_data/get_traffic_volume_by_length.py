from numpy.lib.twodim_base import diagflat
import requests
import pandas as pd

url = 'https://www.vegvesen.no/trafikkdata/api/'
headers =  {"content-type": "application/json"}
query = ''' query {
  trafficData(trafficRegistrationPointId: "44656V72812") {
    volume {
      byDay(
        from: "2021-08-01T12:00:00+02:00"
        to: "2021-08-08T14:00:00+02:00"
      ) {
        edges {
          node {
            from
            to
            byLengthRange{
              lengthRange {
                representation
              }
              total {
              volumeNumbers {
                volume
              }
              coverage {
                percentage
              }
            }
            }
            total {
              volumeNumbers {
                volume
              }
              coverage {
                percentage
              }
            }
          }
        }
      }
    }
  }
}'''

r = requests.post(url, json = {'query':query}, headers = headers)
json_data = r.json()

print(json_data)