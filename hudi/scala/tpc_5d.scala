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
import java.util.Date
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

val dataFileName = sys.env("DATA_FILE")

val layoutOptStrategy = sys.env("HUDI_LAYOUT"); 

val bmcPattern = sys.env("HUDI_BMC_PATTERN");

val bytesNum = sys.env("BYTES_NUM");

val inputPath = s"file:///home/pg/hudi/parquet_files/${dataFileName}.parquet"
val tableName = if (layoutOptStrategy == "bmc") {
  s"${dataFileName}_${layoutOptStrategy}_returnflag_linestatus_BMC"
} else if (layoutOptStrategy == "hilbert" || layoutOptStrategy == "z-order") {
  s"${dataFileName}_${layoutOptStrategy}_returnflag_linestatus_${bytesNum}"
} else {
  s"${dataFileName}_${layoutOptStrategy}_returnflag_linestatus"
}
val outputPath = s"file:///home/pg/hudi/hudi_files/$tableName"


def safeTableName(s: String) = s.replace('-', '_')

val commonOpts =
 Map(
   "hoodie.compact.inline" -> "false",
   "hoodie.bulk_insert.shuffle.parallelism" -> "10"
 )

var df = spark.read.parquet(inputPath)

df = df.withColumn("l_shipdate", col("l_shipdate").cast("date"))
df = df.withColumn("l_commitdate", col("l_commitdate").cast("date"))
df = df.withColumn("l_receiptdate", col("l_receiptdate").cast("date"))

def writeHudiTable(df: DataFrame,
                   tableName: String,
                   outputPath: String,
                   layoutOptStrategy: String,
                   bytesNum: String,
                   commonOpts: Map[String, String]): Unit = {
  println(s"tableName $tableName:")
  println(s"outputPath $outputPath:")
  println(s"layoutOptStrategy $layoutOptStrategy:")
  if (layoutOptStrategy == "default") {
    df.write.format("hudi")
    .option(DataSourceWriteOptions.TABLE_TYPE.key(), COW_TABLE_TYPE_OPT_VAL)
    .option("hoodie.table.name", tableName)
    .option(PRECOMBINE_FIELD.key(), "record_id")
    .option(RECORDKEY_FIELD.key(), "record_id")
    .option(DataSourceWriteOptions.PARTITIONPATH_FIELD.key(), "l_returnflag,l_linestatus")
    .option("hoodie.clustering.inline", "true")
    .option("hoodie.clustering.inline.max.commits", "1")
    // NOTE: Small file limit is intentionally kept _ABOVE_ target file-size max threshold for Clustering,
    // to force re-clustering
    .option("hoodie.clustering.plan.strategy.small.file.limit", String.valueOf(64 * 1024 * 1024)) // 64Mb
    .option("hoodie.clustering.plan.strategy.target.file.max.bytes", String.valueOf(4 * 1024 * 1024)) // 4Mb
    // NOTE: We're increasing cap on number of file-groups produced as part of the Clustering run to be able to accommodate for the
    // whole dataset (~33Gb)
    .option("hoodie.clustering.plan.strategy.max.num.groups", String.valueOf(4096))
    // .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_ENABLE.key, layout_opt)
    // .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_STRATEGY.key, layoutOptStrategy)
    // .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_STRATEGY_BMC.key, bmcPattern)
    // .option(HoodieClusteringConfig.PLAN_STRATEGY_SORT_COLUMNS.key, "l_commitdate,l_receiptdate")
    .option(DataSourceWriteOptions.OPERATION.key(), DataSourceWriteOptions.BULK_INSERT_OPERATION_OPT_VAL)
    .option(BULK_INSERT_SORT_MODE.key(), "NONE")
    .options(commonOpts)
    //  .mode(ErrorIfExists)
    .save(outputPath)
  } else {
    df.write.format("hudi")
    .option(DataSourceWriteOptions.TABLE_TYPE.key(), COW_TABLE_TYPE_OPT_VAL)
    .option("hoodie.table.name", tableName)
    .option(PRECOMBINE_FIELD.key(), "record_id")
    .option(RECORDKEY_FIELD.key(), "record_id")
    .option(DataSourceWriteOptions.PARTITIONPATH_FIELD.key(), "l_returnflag,l_linestatus")
    .option("hoodie.clustering.inline", "true")
    .option("hoodie.clustering.inline.max.commits", "1")
    // NOTE: Small file limit is intentionally kept _ABOVE_ target file-size max threshold for Clustering,
    // to force re-clustering
    .option("hoodie.clustering.plan.strategy.small.file.limit", String.valueOf(64 * 1024 * 1024)) // 64Mb
    .option("hoodie.clustering.plan.strategy.target.file.max.bytes", String.valueOf(4 * 1024 * 1024)) // 4Mb
    // NOTE: We're increasing cap on number of file-groups produced as part of the Clustering run to be able to accommodate for the
    // whole dataset (~33Gb)
    .option("hoodie.clustering.plan.strategy.max.num.groups", String.valueOf(4096))
    .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_ENABLE.key, "true")
    .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_STRATEGY.key, layoutOptStrategy)
    .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_STRATEGY_BMC.key, bmcPattern)
    .option(HoodieClusteringConfig.LAYOUT_OPTIMIZE_STRATEGY_BYTES.key, bytesNum)
    .option(HoodieClusteringConfig.PLAN_STRATEGY_SORT_COLUMNS.key, "l_commitdate,l_receiptdate,l_shipdate,l_linenumber,l_quantity")
    .option(DataSourceWriteOptions.OPERATION.key(), DataSourceWriteOptions.BULK_INSERT_OPERATION_OPT_VAL)
    .option(BULK_INSERT_SORT_MODE.key(), "NONE")
    .options(commonOpts)
    //  .mode(ErrorIfExists)
    .save(outputPath)
  }

}

val path = Paths.get(new URI(outputPath))

if (!Files.exists(path)) {
  writeHudiTable(df, tableName, outputPath, layoutOptStrategy, bytesNum, commonOpts)
} else {
  println("File exists")
}


 //////////////////////////////////////////////////////////////
// Reading
///////////////////////////////////////////////////////////////

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




val spark = SparkSession.builder.appName("MyApp").getOrCreate()
spark.sparkContext.addSparkListener(new MyCustomListener())

val outputFile = if (layoutOptStrategy == "bmc") {
  s"./hudi/result/${dataFileName}_${layoutOptStrategy}_5d.json"
} else {
  s"./hudi/result/${dataFileName}_${layoutOptStrategy}_5d_${bytesNum}.json"
}

// val outputFile = s"./hudi/result/${dataFileName}_${layoutOptStrategy}.json"
val fileOut = new PrintStream(new FileOutputStream(outputFile, false))
System.setOut(fileOut)




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

fileOut.println("{")

fileOut.println(s""""layout": "$layoutOptStrategy",""")
fileOut.println(s""""query num": "${Constants.NumQueries}",""")
fileOut.println(s""""name": "$dataFileName",""")
fileOut.println(s""""measurements": [""")

// runQuery("Q1", false,
//     stageMetrics.runAndMeasure {
//     val gap = random.nextInt(6)
//     val query = s"""
//     select
//       l_returnflag,
//       l_linestatus,
//       sum(l_quantity) as sum_qty,
//       sum(l_extendedprice) as sum_base_price,
//       sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
//       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
//       avg(l_quantity) as avg_qty,
//       avg(l_extendedprice) as avg_price,
//       avg(l_discount) as avg_disc,
//       count(*) as count_order
//     from
//       $rawSnapshotTableName
//     where
//       l_shipdate <= add_months(date '1998-11-28', $gap)
//     group by
//       l_returnflag,
//       l_linestatus
//     order by
//       l_returnflag,
//       l_linestatus
//     """
//     println(query)
//       spark.sql(query).show()
//     })

// runQuery("Q2", false,
//       stageMetrics.runAndMeasure {
//       val gap = random.nextInt(6) + 9
//       val quantity = random.nextInt(10)
//       val query = s"""
//       select
//         sum(l_extendedprice * l_discount) as revenue
//       from
//         $rawSnapshotTableName
//       where
//         l_shipdate >= '1992-08-01'
//         and l_shipdate < add_months(date '1992-08-01', $gap)
//         and l_returnflag = 'R'
//         and l_discount between 0.02 and 0.1
//         and l_quantity < $quantity;
//       """
//       println(query)
//       spark.sql(query).show()
//     })

// runQuery("Q3", false,
//       stageMetrics.runAndMeasure {
//       val gap = random.nextInt(6) + 9
//       val query = s"""
//       select
//         l_shipmode,
//         count(*) as count_item
//       from
//         $rawSnapshotTableName
//       where
//         l_linestatus = 'F' and
//         l_shipdate < l_receiptdate and
//         l_commitdate < l_receiptdate
//         and l_shipdate < l_commitdate
//         and l_receiptdate >= date '1997-12-01'
//         and l_receiptdate < add_months(date '1997-12-01', $gap)
//       group by
//         l_shipmode
//       order by
//         l_shipmode;
//       """
//       println(query)
//       spark.sql(query).show()
//     })

// runQuery("Q4", false,
//       stageMetrics.runAndMeasure {
//       val gap = random.nextInt(4)
//       val query = s"""
//       select
//         100.00 * sum(case
//           when l_shipmode like 'AIR%'
//             then l_extendedprice * (1 - l_discount)
//           else 0
//         end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
//       from
//         $rawSnapshotTableName
//       where
//         l_returnflag = 'N' and
//         l_commitdate >= '1997-11-01'
//         and l_commitdate < add_months(date '1997-12-01', $gap);
//       """
//       println(query)
//       spark.sql(query).show()
//     })

runQuery("Q5", true,
      stageMetrics.runAndMeasure {
      val gap = random.nextInt(60) + 12
      val linenumber_min = random.nextInt(5)
      val linenumber_max = linenumber_min + random.nextInt(5)
      val quantity_max = random.nextInt(10)

      val query = s"""
      select
        l_commitdate,
        sum(l_extendedprice) as total_daily
      from
        $rawSnapshotTableName
      where
        l_returnflag = 'A'
        AND
        l_commitdate >= '1991-01-01'
        and l_commitdate < add_months(date '1991-02-01', $gap)
        and l_shipdate < add_months(date '1991-02-01', $gap)
        and l_receiptdate < add_months(date '1991-02-01', $gap)
        and l_linenumber > $linenumber_min
        and l_linenumber < $linenumber_max
        and l_quantity < $quantity_max
      group by
        l_commitdate
      order by
        l_commitdate;
      """
      println(query)
      spark.sql(query).show()
    })


fileOut.println("]}")

fileOut.close()

