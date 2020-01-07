package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession

object StringIndexerApp extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("StringIndexerApp")
    .master("local[*]").getOrCreate()

  val df = spark.createDataFrame(Seq(
    (0, "Jason", "Germany"),
    (1, "David", "France"),
    (2, "Martin", "Spain"),
    (3, "Jason", "USA"),
    (4, "Daiel", "UK"),
    (5, "Samsung", "India"),
    (6, "David", "Ireland"),
    (7, "Jason", "Netherlands")
  )).toDF("id", "name", "address")

  val indexer = new StringIndexer()
    .setInputCol("name")
    .setOutputCol("label")
    .fit(df)

  // the most frequent label gets index 0
  // if the input column is numeric we cast it to string and index the string values
  val indexed = indexer.transform(df)

  indexed.show(false)
}
