query = ''' query {
  trafficData(trafficRegistrationPointId: "44656V72812") {
    volume {
      byHour(
        from: "2019-10-24T12:00:00+02:00"
        to: "2019-10-24T14:00:00+02:00"
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