import org.apache.hadoop.fs.{FileStatus, Path}
import scala.collection.JavaConversions._
import org.apache.spark.sql.SaveMode._
import org.apache.hudi.{DataSourceReadOptions, DataSourceWriteOptions}
import org.apache.hudi.DataSourceWriteOptions._
import org.apache.hudi.common.fs.FSUtils
import org.apache.hudi.common.table.HoodieTableMetaClient
import org.apache.hudi.common.util.ClusteringUtils
import org.apache.hudi.config.HoodieClusteringConfig
import org.apache.hudi.config.HoodieWriteConfig._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

import java.io.{File, FileOutputStream, PrintStream}
import java.nio.file.{Files, Paths}
import java.net.URI
import java.util.stream.Collectors
import scala.util.Random


import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import org.apache.spark.sql.SparkSession

import org.apache.spark.scheduler.SparkListenerEvent

import org.apache.spark.scheduler.{SparkListener, SparkListenerTaskEnd, SparkListenerExecutorMetricsUpdate, SparkListenerJobStart, SparkListenerJobEnd}

import scala.collection.mutable

import org.apache.spark.sql.execution.ui.{SparkListenerSQLExecutionStart, SparkListenerSQLExecutionEnd, SparkListenerDriverAccumUpdates}

object Global {
  @volatile var readSizeGlobal: Double = 0.0 
}

object Constants {
  lazy val NumQueries: Int = sys.env.get("NUM_QUERIES") match {
    case Some(value) => value.toInt  // Try to get the value from the environment variable and convert it to Int
    case None => 50  // Default value if the environment variable is not set
  }
}

val seed = 12345L
val random = new Random(seed)

class MyCustomListener extends SparkListener {

  private val jobStartTimes = mutable.HashMap[Int, Long]()

  // override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
  //   // start recording
  //   jobStartTimes(jobStart.jobId) = jobStart.time
  //   // println(s"Job ${jobStart.jobId} start from ${jobStart.time}")
  // }

  // override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {
  //   // get duration time
  //   val startTime = jobStartTimes.getOrElse(jobEnd.jobId, jobEnd.time)
  //   val duration = jobEnd.time - startTime
  //   println(s"Job ${jobEnd.jobId} end at ${jobEnd.time}, duration: $duration ms")
  // }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case driverAccumUpdates: SparkListenerDriverAccumUpdates =>

        val length = driverAccumUpdates.accumUpdates.length
        if (length == 5) {
          val valuesList = driverAccumUpdates.accumUpdates.sortBy(_._1).map(_._2)

          val fileRead = valuesList(0)
          val readSize = valuesList(2) / (1024.0 * 1024)
          // valuesList(1) // metadata time total 
          //  valuesList(4) // dynamic partition pruning time total
          val formattedReadSize = f"$readSize%.2f"
          Global.readSizeGlobal += readSize
          // System.out.println(s"""{"file_read": "$fileRead", "IO": "$formattedReadSize",""")
          // println("valuesList contains:")
          // valuesList.foreach(value => println(value))
          
        }
      case _ =>
        println("other event")
    }
  }
}


val layoutOptStrategy = sys.env("HUDI_LAYOUT"); 
val bytesNum = sys.env("BYTES_NUM");

// val layoutOptStrategy = "bmc"

val bmcPattern = sys.env("HUDI_BMC_PATTERN");
// val bmcPattern = "AAAAAAAABBBBBBBB";

val inputPath = s"file:///home/pg/hudi/parquet_files/nyc.parquet"
val tableName = if (layoutOptStrategy == "bmc") {
  s"nyc_${layoutOptStrategy}_passenger_count_${bmcPattern}"
} else if (layoutOptStrategy == "hilbert" || layoutOptStrategy == "z-order") {
  s"nyc_${layoutOptStrategy}_passenger_count_${bytesNum}"
} else {
  s"nyc_${layoutOptStrategy}_passenger_count"
}
val outputPath = s"file:///home/pg/hudi/hudi_files/$tableName"


def safeTableName(s: String) = s.replace('-', '_')

val commonOpts =
 Map(
   "hoodie.compact.inline" -> "false",
   "hoodie.bulk_insert.shuffle.parallelism" -> "10"
 )

val df = spark.read.parquet(inputPath)

// VendorID	tpep_pickup_datetime	tpep_dropoff_datetime	passenger_count	trip_distance	RatecodeID	store_and_fwd_flag	PULocationID	DOLocationID	payment_type	fare_amount	extra	mta_tax	tip_amount	tolls_amount	improvement_surcharge	total_amount


def writeHudiTable(df: DataFrame,
                   tableName: String,
                   outputPath: String,
                   layoutOptStrategy: String,
                   bytesNum: String,
                   commonOpts: Map[String, String]): Unit = {
  val layout_opt = if (layoutOptStrategy == "default") {
    "false"
  } else {
    "true"
  }
  df.write.format("hudi")
  .option(DataSourceWriteOptions.TABLE_TYPE.key(), COW_TABLE_TYPE_OPT_VAL)
  .option("hoodie.table.name", tableName)
  .option(PRECOMBINE_FIELD.key(), "record_id")
  .option(RECORDKEY_FIELD.key(), "record_id")
  .option(DataSourceWriteOptions.PARTITIONPATH_FIELD.key(), "passenger_count") // TODO partition by PULocationID
  .option("hoodie.clustering.inline", "true")
  .option("hoodie.clustering.inline.max.commits", "1")
  // NOTE: Small file limit is intentionally kept _ABOVE_ target file-size max threshold for Clustering,
  // to force re-clustering
  .option("hoodie.clustering.plan.strategy.small.file.limit", String.valueOf(256 * 1024 * 1024)) // 64Mb
  .option("hoodie.clustering.plan.strategy.target.file.max.bytes", String.valueOf(64 * 1024 * 1024)) // 4Mb
  // NOTE: We're increasing cap on number of file-groups produced as part of the Clustering run to be able to accommodate for the
  // whole dataset (~33Gb)
  .option("hoodie.clustering.plan.strategy.max.num.groups", String.valueOf(4096))
  .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_ENABLE.key, layout_opt)
  .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_STRATEGY.key, layoutOptStrategy)
  .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_STRATEGY_BMC.key, bmcPattern)
  .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_STRATEGY_BYTES.key, bytesNum)
  .option(HoodieClusteringConfig.PLAN_STRATEGY_SORT_COLUMNS.key, "PULocationID,DOLocationID")
  .option(DataSourceWriteOptions.OPERATION.key(), DataSourceWriteOptions.BULK_INSERT_OPERATION_OPT_VAL)
  .option(BULK_INSERT_SORT_MODE.key(), "NONE")
  .options(commonOpts)
  //  .mode(ErrorIfExists)
  .save(outputPath)

}

val path = Paths.get(new URI(outputPath))

if (!Files.exists(path)) {
  writeHudiTable(df, tableName, outputPath, layoutOptStrategy, bytesNum, commonOpts)
} else {
  println("File exists")
}


// Temp Table w/ Data Skipping DISABLED
val readDf: DataFrame =
  spark.read.option(DataSourceReadOptions.ENABLE_DATA_SKIPPING.key(), "false").format("hudi").load(outputPath)

val rawSnapshotTableName = safeTableName(s"${tableName}_sql_snapshot")

readDf.createOrReplaceTempView(rawSnapshotTableName)


// Temp Table w/ Data Skipping ENABLED
val readDfSkip: DataFrame =
  spark.read.option(DataSourceReadOptions.ENABLE_DATA_SKIPPING.key(), "true").format("hudi").load(outputPath)

val dataSkippingSnapshotTableName = safeTableName(s"${tableName}_sql_snapshot_skipping")

readDfSkip.createOrReplaceTempView(dataSkippingSnapshotTableName)

val stageMetrics = ch.cern.sparkmeasure.StageMetrics(spark)


System.out.println("--------------------------------------")
System.out.println(s"layout: ${layoutOptStrategy}")


val spark = SparkSession.builder.appName("MyApp").getOrCreate()
spark.sparkContext.addSparkListener(new MyCustomListener())

val outputFile = if (layoutOptStrategy == "bmc") {
  s"./hudi/result/nyc_${layoutOptStrategy}_${bmcPattern}.json"
} else {
  s"./hudi/result/nyc_${layoutOptStrategy}_${bytesNum}.json"
}

// val outputFile = s"./hudi/results/nyc_${layoutOptStrategy}.json"
val fileOut = new PrintStream(new FileOutputStream(outputFile, false))
System.setOut(fileOut)

fileOut.println("{")

fileOut.println(s""""layout": "$layoutOptStrategy",""")
fileOut.println(s""""query num": "${Constants.NumQueries}",""")
fileOut.println(s""""name": "nyc",""")
fileOut.println(s""""measurements": [""")


def runQuery(queryName: String, isLast: Boolean, executeQuery: => Unit) = {
  println(s"runQuery $queryName:")

  var totalBytesRead: Long = 0L
  var totalElapsedTime: Long = 0L
  var totalExecutorRunTime: Long = 0L
  var totalExecutorCpuTime: Long = 0L

  for (_ <- 1 to Constants.NumQueries) {
    stageMetrics.runAndMeasure {
      executeQuery
    }
    
    val metrics = stageMetrics.aggregateStageMetrics()
    val bytesRead: Long = metrics.get("bytesRead").getOrElse(0L) 
    val elapsedTime: Long = metrics.get("elapsedTime").getOrElse(0L) 
    val executorRunTime: Long = metrics.get("executorRunTime").getOrElse(0L) 
    val executorCpuTime: Long = metrics.get("executorCpuTime").getOrElse(0L) 

    totalBytesRead += bytesRead
    totalElapsedTime += elapsedTime
    totalExecutorRunTime += executorRunTime
    totalExecutorCpuTime += executorCpuTime
  }
  val avgBytesRead = totalBytesRead / Constants.NumQueries
  val avgElapsedTime = totalElapsedTime / Constants.NumQueries
  val avgExecutorRunTime = totalExecutorRunTime / Constants.NumQueries
  val avgExecutorCpuTime = totalExecutorCpuTime / Constants.NumQueries
  Global.readSizeGlobal /= Constants.NumQueries
  val avgReadSize = f"${Global.readSizeGlobal}%.2f"
    if (isLast) {
    fileOut.println(s"""{"query": "$queryName", "cpuTime": $avgExecutorCpuTime, "IO":$avgReadSize}""")
  } else {
    fileOut.println(s"""{"query": "$queryName", "cpuTime": $avgExecutorCpuTime, "IO":$avgReadSize},""")
  } 
  println(s"""{"query": "$queryName", "runTime": $avgExecutorCpuTime, "IO":$avgReadSize}""")
  Global.readSizeGlobal = 0
}

runQuery("Q1", false,
    stageMetrics.runAndMeasure {
    val gap = random.nextInt(6) + 6
    val randomPULower = random.nextInt(100) 
    val randomPUUpper = randomPULower + 100
    val randomDOLower = random.nextInt(100)
    val randomDOUpper = randomDOLower + 100
    val query = s"""
    SELECT
    PULocationID,
    DOLocationID,
    COUNT(*) trips
    FROM
      $rawSnapshotTableName
    WHERE
      TO_TIMESTAMP(tpep_dropoff_datetime) BETWEEN '2017-01-01' 
      AND add_months(date '2017-01-01', $gap)
      AND passenger_count > 2
      AND PULocationID BETWEEN $randomPULower AND $randomPUUpper
      AND DOLocationID BETWEEN $randomDOLower AND $randomDOUpper
    GROUP BY
      PULocationID,
      DOLocationID
    """
    println(query)
      spark.sql(query).show()
    })

runQuery("Q2", false,
      stageMetrics.runAndMeasure {
      val randomPassengerCount = random.nextInt(5) + 2
      val randomPULower = random.nextInt(100) + 100
      val randomDOLower = random.nextInt(100)
      val query = s"""
      SELECT
        PULocationID,
        DOLocationID,
        COUNT(*) trips
      FROM
        $rawSnapshotTableName
      WHERE
        trip_distance > 0
        AND passenger_count > $randomPassengerCount
        AND fare_amount / trip_distance BETWEEN 2 AND 10
        AND tpep_dropoff_datetime > tpep_pickup_datetime
        AND PULocationID < $randomPULower
        AND DOLocationID < $randomDOLower
      GROUP BY
        PULocationID,
        DOLocationID
      """
      println(query)
      spark.sql(query).show()
    })

runQuery("Q3", false,
      stageMetrics.runAndMeasure {
      val randomPassengerCount = random.nextInt(5) + 2
      val randomPULower = random.nextInt(100)
      val randomDOLower = random.nextInt(100) + 100
      val query = s"""
      SELECT
        PULocationID,
        DOLocationID,
        COUNT(*) trips
      FROM
        $rawSnapshotTableName
      WHERE
        trip_distance > 0
        AND passenger_count > $randomPassengerCount
        AND fare_amount / trip_distance BETWEEN 2 AND 10
        AND TO_TIMESTAMP(tpep_dropoff_datetime) > TO_TIMESTAMP(tpep_pickup_datetime)
        AND PULocationID < $randomPULower
        AND DOLocationID < $randomDOLower
      GROUP BY
        PULocationID,
        DOLocationID
      """
      println(query)
      spark.sql(query).show()
    })

runQuery("Q4", false,
      stageMetrics.runAndMeasure {
      val randomPassengerCount = random.nextInt(5) + 3
      val randomPULower = random.nextInt(100)
      val randomDOLower = random.nextInt(100)
      val gap = random.nextInt(6) + 1
      val query = s"""
      SELECT
        PULocationID,
        COUNT(*) trips
      FROM
        $rawSnapshotTableName
      WHERE
        TO_TIMESTAMP(tpep_pickup_datetime) BETWEEN '2017-01-01' AND add_months(date '2017-01-01', $gap)
        AND
        TO_TIMESTAMP(tpep_dropoff_datetime) > TO_TIMESTAMP(tpep_pickup_datetime)
        AND passenger_count = $randomPassengerCount
        AND
        PULocationID < $randomPULower
        AND
        DOLocationID < $randomDOLower
      GROUP BY
        PULocationID
      """
      println(query)
      spark.sql(query).show()
    })

runQuery("Q5", true,
      stageMetrics.runAndMeasure {
      val randomPULower = random.nextInt(100) 
      val randomPUUpper = randomPULower + 20 
      val randomDOLower = random.nextInt(100)
      val randomDOUpper = randomDOLower + 20
      val randomPassengerCount = random.nextInt(5) + 2

      val query = s"""
        SELECT
          COUNT(*) trips
        FROM
          $rawSnapshotTableName
        WHERE
          PULocationID BETWEEN $randomPULower AND $randomPUUpper
          AND DOLocationID BETWEEN $randomDOLower AND $randomDOUpper
          AND passenger_count > $randomPassengerCount
      """
      println(query)
      spark.sql(query).show()
    })


fileOut.println("]}")

fileOut.close()

