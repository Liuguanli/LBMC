import java.io.{PrintStream, FileOutputStream}
import scala.util.Random

val seed = 12345L
val random = new Random(seed)
object Constants {
  lazy val NumQueries: Int = sys.env.get("NUM_QUERIES") match {
    case Some(value) => value.toInt  // Try to get the value from the environment variable and convert it to Int
    case None => 50  // Default value if the environment variable is not set
  }
}
def runQuery(queryName: String, isLast: Boolean, fileOut: PrintStream) = {
  val random = new Random()
  println(s"runQuery $queryName:")

  val rangeSettings = queryName match {
    case "Q1" => (250, 250, 50, 50)
    case "Q2" => (200, 200, 100, 10)
    case "Q3" => (200, 200, 10, 100)
    case "Q4" => (200, 200, 20, 20)
    case "Q5" => (200, 200, 0, 0)

    case _    => (100, 200, 20, 20) // Default range
  }

  val results = for (_ <- 1 to Constants.NumQueries) yield {
    val randomPULower = random.nextInt(rangeSettings._1)
    val randomPUUpper = randomPULower + rangeSettings._3
    val randomDOLower = random.nextInt(rangeSettings._2)
    val randomDOUpper = randomDOLower + rangeSettings._4
    
    if (queryName == "Q4") {
      // Special handling for Q4
      val adjustedPULower = random.nextInt(10)
      val adjustedDOLower = random.nextInt(10)
      val adjustedPUUpper = randomPUUpper
      val adjustedDOUpper = randomDOUpper
      List(adjustedPULower, adjustedDOLower, adjustedPUUpper, adjustedDOUpper)
    } else {
      List(randomPULower, randomDOLower, randomPUUpper, randomDOUpper)
    }
  }

  // Manually construct JSON
  val jsonResults = results.map(r => s"[${r.mkString(", ")}]").mkString(", ")
  val json = s""""$queryName": [$jsonResults]"""

  // Write the JSON string to the file
  fileOut.println(json)

  if (isLast) {
    fileOut.println("}")
    fileOut.close()
  } else {
    fileOut.println(",")
  }
}

// Example usage
val fileOut = new PrintStream(new FileOutputStream("./hudi/result/nyc_query.json", false))
fileOut.println("{")
runQuery("Q1", false, fileOut)
runQuery("Q2", false, fileOut)
runQuery("Q3", false, fileOut)
runQuery("Q4", false, fileOut)
runQuery("Q5", true, fileOut)
