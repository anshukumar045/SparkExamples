package project1.machinelearning

import org.apache.spark.sql.SparkSession

package object AnalyzingInsurance {

  val spark = SparkSession.builder().appName("AnalyzingInsuranceApp")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  val train = "data/allstate-claims-severity/train.csv"

  val trainInput = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv")
    .load(train)
    .cache

  val test = "data/allstate-claims-severity/test.csv"
  val testInput = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv")
    .load(test)
    .cache
}
