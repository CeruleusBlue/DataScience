<span>
    <h5>The cell below: </h5>
    <ul>
        <li>Reads the data from the file onto a spark dataframes</li>
        <br>
        <li>Separates the uris and hyperlinks(dropping the rdf prefixes) into two dataframes</li>
        <br>
        <li>Creates views for the dataframes so that sql queries can be performed on them</li>
    </ul>
</span>

```py
import pyspark, pyspark.sql, numpy as np

spark:pyspark.sql.SparkSession = pyspark.sql.SparkSession.Builder().getOrCreate()

df = spark.read.csv('Task2/gr0.California', sep=" ")
uri = df.filter("_c0 = 'n'").drop('_c0')
hyperlink = df.filter("_c0 = 'e'").drop('_c0')
uri.createGlobalTempView('uri')
hyperlink.createGlobalTempView('hyperlink')
del df
```

<span>
    <h5>The cell below: </h5>
    <li>Runs an sql query that returns all uris and the out-degree for rows which contain the maximum out-degree</li>
    <br>
    <li>This has been done using nested select queries and joins</li>
</span>

```py
maxOut = spark.sql(
    """
    SELECT global_temp.uri._c2 AS url, count
    FROM(
        SELECT _c1, count(_c1) AS count
        FROM global_temp.hyperlink
        GROUP BY _c1
        ) T1
    RIGHT JOIN(
        SELECT max(count) as maxCount
        FROM(
            SELECT _c1, count(_c1) AS count
            FROM global_temp.hyperlink
            GROUP BY _c1
            ) T1
        )T2
    ON count = maxCount
    LEFT JOIN global_temp.uri
    ON T1._c1 = global_temp.uri._c1
    """
)
maxOut.show(truncate=False)
```
```
+------------------------------------------+-----+
|url                                       |count|
+------------------------------------------+-----+
|http://www.water.ca.gov/www.gov.sites.html|164  |
+------------------------------------------+-----+
```
<span style="color:rgb(0, 162, 255)">
    The query above outputs only one row which contains 
    the URI for the website and the out-degree.
</span>
<span>
    <h5>The cell below: </h5>
    <li>Runs an sql query that returns all uris and the in-degree for rows which contain the maximum in-degree</li>
    <br>
    <li>This has been done using nested select queries and joins</li>
</span>

```py
maxIn = spark.sql(
    """
    SELECT global_temp.uri._c2 AS url, count
    FROM(
        SELECT _c2, count(_c2) AS count
        FROM global_temp.hyperlink
        GROUP BY _c2
        ) T1
    RIGHT JOIN(
        SELECT max(count) as maxCount
        FROM(
            SELECT _c2, count(_c2) AS count
            FROM global_temp.hyperlink
            GROUP BY _c2
            ) T1
        )T2
    ON count = maxCount
    LEFT JOIN global_temp.uri
    ON T1._c2 = global_temp.uri._c1
    """
)
maxIn.show(truncate=False)
```
```
+---------------------+-----+
|url                  |count|
+---------------------+-----+
|http://www.yahoo.com/|199  |
+---------------------+-----+
```
<span style="color:rgb(0, 162, 255)">
    The query above outputs only one row which contains 
    the URI for the website and the in-degree.
</span>
<span>
    <h5>The cell below: </h5>
    <li>Runs an sql query to calculate the average out-degree</li>
    <br>
    <li>This has been done using a nested select query</li>
</span>

```py
meanOut = spark.sql(
    """
    SELECT avg(count) AS outAverage
    FROM(
        SELECT _c1, count(_c1) AS count
        FROM global_temp.hyperlink
        GROUP BY _c1
        ) T1
    """
)
meanOut.show(truncate=False)
```
```
+-----------------+
|outAverage       |
+-----------------+
|3.212651680923016|
+-----------------+
```
<span style="color:rgb(0, 162, 255)">
    The query above outputs only one row which contains 
    the average out-degree value.
</span>
<span>
    <h5>The cell below: </h5>
    <li>Runs an sql query to calculate the average in-degree</li>
    <br>
    <li>This has been done using a nested select query</li>
</span>

```py
meanIn = spark.sql(
    """
    SELECT avg(count) AS inAverage
    FROM(
        SELECT _c2, count(_c2) AS count
        FROM global_temp.hyperlink
        GROUP BY _c2
        ) T1
    """
)
meanIn.show(truncate=False)
```
```
+-----------------+
|inAverage        |
+-----------------+
|7.694140066698428|
+-----------------+
```
<span style="color:rgb(0, 162, 255)">
    The query above outputs only one row which contains 
    the average in-degree value.
</span>
<span>
    <h5>The cell below: </h5>
    <li>Runs an sql query to return the count of rows which has an out-degree of 0</li>
    <br>
    <li>This has been done using a nested select query.</li>
    <br>
    <li>The query is based on the fact that if the uri doesn't have any out hyperlink entries 
        then the out-degree of the uri is 0.</li>
    <br>
</span>

```py
zeroOut = spark.sql(
    """
    SELECT count(_c1) AS webpageCount
    FROM global_temp.uri
    WHERE _c1 NOT IN(
        SELECT DISTINCT _c1
        FROM global_temp.hyperlink
    )
    """
)
zeroOut.show(truncate=False)
```
```
+------------+
|webpageCount|
+------------+
|4637        |
+------------+
```
<span style="color:rgb(0, 162, 255)">
    The query above outputs only one row which contains 
    the count of the uris with zero out-degree.
</span>