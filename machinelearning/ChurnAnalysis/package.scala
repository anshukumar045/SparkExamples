package project1.machinelearning

import org.apache.spark.sql.{Dataset, SparkSession}

package object ChurnAnalysis {

  val spark = SparkSession.builder().appName("ChurnAnalysis")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  import spark.implicits._
  import DataPreparation._
  val trainSet: Dataset[CustomerAccount] = spark.read
    .option("inferSchema", "true")
    .format("csv")
    .schema(schema)
    .load("data/churn-bigml-80.csv").na.drop
    .as[CustomerAccount]

  trainSet.cache
  // see related statistics of the training set using the describe()
  /*
  val stateDF = trainSet.describe()
  stateDF.show(false) */

  // variable correlation with churn
  trainSet.groupBy("churn").sum("total_international_num_calls").show(false)
  trainSet.groupBy("churn").sum("total_international_charge").show()

  val testSet: Dataset[CustomerAccount] = spark.read
    .option("inferSchema", "true")
    .format("csv")
    .schema(schema)
    .load("data/churn-bigml-20.csv").na.drop
    .as[CustomerAccount]

  testSet.cache


}
