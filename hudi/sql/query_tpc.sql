-- SQL 1
SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM
    lineitem
WHERE
    l_shipdate <= ADD_MONTHS(DATE '1998-11-28', ':1')
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;

-- SQL 6
SELECT
    SUM(l_extendedprice * l_discount) AS revenue
FROM
    lineitem
WHERE
    l_shipdate >= '1992-08-01'
    AND l_shipdate < ADD_MONTHS(DATE '1992-08-01', ':1')
    AND l_returnflag = 'R'
    AND l_discount BETWEEN 0.02 AND 0.1
    AND l_quantity < ':2';

-- SQL 12
SELECT
    l_shipmode,
    COUNT(*) AS count_item
FROM
    lineitem
WHERE
    l_linestatus = 'F'
    AND l_commitdate < l_receiptdate
    AND l_shipdate < l_commitdate
    AND l_receiptdate >= DATE '1997-12-01'
    AND l_receiptdate < ADD_MONTHS(DATE '1997-12-01', ':1')
GROUP BY
    l_shipmode
ORDER BY
    l_shipmode;

-- SQL 14
SELECT
    100.00 * SUM(CASE
        WHEN l_shipmode LIKE 'AIR%'
        THEN l_extendedprice * (1 - l_discount)
        ELSE 0
    END) / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM
    lineitem
WHERE
    l_returnflag = 'N'
    AND l_commitdate >= '1997-11-01'
    AND l_commitdate < ADD_MONTHS(DATE '1997-12-01', ':1');

-- SQL 17
SELECT
    l_commitdate,
    SUM(l_extendedprice) AS total_daily
FROM
    lineitem
WHERE
    l_returnflag = 'A'
    AND l_commitdate >= '1991-01-01'
    AND l_commitdate < ADD_MONTHS(DATE '1991-02-01', ':1')
GROUP BY
    l_commitdate
ORDER BY
    l_commitdate;
