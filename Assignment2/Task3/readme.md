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
import pyspark, pyspark.sql 
from pyspark.sql.functions import *  
import numpy as np

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
    <ul>
        <li>Runs an sql query which returns the out-degree for each distinct value of uri present in the out link column of hyperlink.</li>
    </ul>
</span>

```py
outLinkCount = spark.sql(
    """
    SELECT _c1 AS link, count(*) AS count
    FROM global_temp.hyperlink
    GROUP BY _c1
    ORDER BY count DESC
    """
)
```

```
DataFrame[link: string, count: bigint]
```
<span>
    <h5>The cell below: </h5>
    <ul>
        <li>Stores the dimension of the transition matrix, the dimension of the Block Matrix, and the dimension of each block of the block matrix</li>
        <br>
        <li>Maps the inlink, the outlink and the transition probability into an rdd for the creation of the transition block matrix</li>
        <br>
        <li>Creates a coordinate matrix with the aforementioned rdd and x and y dimensions equal to the dimension of the transition matrix which is then converted into a block matrix with dimension of each block passed as x and y dimensions</li>
        <br>
        <li>Maps the id column of the uri into an rdd with id as row index, 0 as the column index and probability of the id as the matrix entry </li>
        <br>
        <li>Creates a coordinate matrix with the aforementioned rdd and x dimension equal to the dimension of the transition matrix and y equal to 1 which is then converted into a block matrix with dimension of each block and 1 passed as x and y dimensions respectively</li>
    </ul>
</span>

```py
from pyspark.mllib.linalg import *
from pyspark.mllib.linalg.distributed import *

#dimension of transition matrix
matrixSize = uri.count()
#dimension of block matrix
totalBlocks = 5
#dimension of each block of the block matrix
blockSize = int(np.ceil(matrixSize/totalBlocks))

transRdd = hyperlink.withColumnRenamed('_c1','link')\
            .join(outLinkCount, on='link', how='full')\
            .rdd.map(lambda x: 
                [x.__getitem__('_c2'),
                x.__getitem__('link'),
                1/x.__getitem__('count')])
                
transMatrix = CoordinateMatrix(transRdd.map(lambda x: MatrixEntry(*x)), matrixSize, matrixSize)\
    .toBlockMatrix(blockSize,blockSize)

rankRdd = uri.select('_c1').rdd.map(lambda x: [int(x.__getitem__(0)), 0, 1/matrixSize])

rankMatrix = CoordinateMatrix(rankRdd.map(lambda x: MatrixEntry(*x)), matrixSize, 1)\
                .toBlockMatrix(blockSize, 1)
```

<span>
    <h5>The cell below: </h5>
    <span style="font-weight:700">Iterates for a specific number of times. Within each iteration:</span>
    <ul>
        <li>The transition matrix is multiplied by the rank matrix with dimensions of the resulting matrix equal to that of the rank matrix</li>
        <br>
        <li>Each entry of the rank block matrix is mapped into an double dimensional array which results in an rdd with two columns. The first contains the matrix row and column location and the second which contains the aforementioned array.</li>
        <br>
        <li>The data is made resilient against spider traps with the &beta; value </li>
        <br>
        <li>The sum of all entries within the block matrix is calculated.</li>
        <br>
        <li>If the sum is zero the loop terminates</li>
        <br>
        <li>The rdd is converted into a Block Matrix again by mapping the array column of the rdd into a Dense Matrix with x equal to the length of the array and y equal to 1</li>
    </ul>
</span>

```py 
ITERATIONS = 10
BETA = 0.85

for i in range(ITERATIONS):
    rankMatrix = transMatrix.multiply(rankMatrix)
    rankRddIter = rankMatrix.blocks.map(lambda x: (x[0], x[1].toArray()))
    rankRddIter = rankRddIter.map(lambda x: (x[0], x[1]*BETA + (1-BETA)/matrixSize))
    tol = rankRddIter.map(lambda x: np.sum(x[1])).reduce(lambda x,y : x+y)
    if tol > 0:
        rankRddIter = rankRddIter.map(lambda x: (x[0], x[1]/tol))
    else:
        print("rank vector sum is 0")
        break
    rankMatrix = BlockMatrix(
        rankRddIter.map(lambda x: (x[0], DenseMatrix(len(x[1]),1,x[1]))), 
        blockSize, 1)
rankMatrix.toCoordinateMatrix().entries.toDF().select('i','value').show(truncate=False)
```
```
+---+---------------------+
|i  |value                |
+---+---------------------+
|0  |0.009137321754279665 |
|1  |0.0020291038525524067|
|2  |5.097803375233102E-5 |
|3  |9.829166697808767E-4 |
|4  |4.502481223333973E-5 |
|5  |1.3177583864833897E-4|
|6  |0.010668038246773488 |
|7  |2.077105134086428E-5 |
|8  |8.193277232323944E-4 |
|9  |0.0012574389224513387|
|10 |0.005354946580873862 |
|11 |2.2228530893132523E-4|
|12 |2.743296629456242E-4 |
|13 |2.077105134086428E-5 |
|14 |9.503722374305101E-4 |
|15 |2.1444314180811653E-4|
|16 |1.4416432606211525E-4|
|17 |0.014369661631170653 |
|18 |6.121578086266045E-5 |
|19 |5.121200128126079E-4 |
+---+---------------------+
only showing top 20 rows
```