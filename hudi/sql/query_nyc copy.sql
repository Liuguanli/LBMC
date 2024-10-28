-- sql 1
SELECT
    PULocationID,
    DOLocationID,
    COUNT(*) AS trips
FROM
    nyc
WHERE
    TO_TIMESTAMP(tpep_dropoff_datetime) BETWEEN '2017-01-01' 
    AND ADD_MONTHS(DATE '2017-01-01', ':1')
    AND passenger_count > 2
GROUP BY
    PULocationID,
    DOLocationID;


-- sql 2
SELECT
    PULocationID,
    DOLocationID,
    COUNT(*) AS trips
FROM
    nyc
WHERE
    trip_distance > 0
    AND passenger_count > ':1'
    AND fare_amount / trip_distance BETWEEN 2 AND 10
    AND tpep_dropoff_datetime > tpep_pickup_datetime
GROUP BY
    PULocationID,
    DOLocationID;


-- sql 3
SELECT
    PULocationID,
    DOLocationID,
    COUNT(*) AS trips
FROM
    nyc
WHERE
    trip_distance > 0
    AND passenger_count > ':1'
    AND fare_amount / trip_distance BETWEEN 2 AND 10
    AND TO_TIMESTAMP(tpep_dropoff_datetime) > TO_TIMESTAMP(tpep_pickup_datetime)
GROUP BY
    PULocationID,
    DOLocationID;


-- sql 4
SELECT
    PULocationID,
    COUNT(*) AS trips
FROM
    nyc
WHERE
    TO_TIMESTAMP(tpep_pickup_datetime) BETWEEN '2017-01-01' AND ADD_MONTHS(DATE '2017-01-01', ':1')
    AND TO_TIMESTAMP(tpep_dropoff_datetime) > TO_TIMESTAMP(tpep_pickup_datetime)
    AND passenger_count = ':2'
    AND PULocationID < ':3'
    AND DOLocationID < ':4'
GROUP BY
    PULocationID;


-- sql 5
SELECT
    COUNT(*) AS trips
FROM
    nyc
WHERE
    PULocationID BETWEEN ':1' AND ':2'
    AND DOLocationID BETWEEN ':3' AND ':4'
    AND passenger_count > ':5';
