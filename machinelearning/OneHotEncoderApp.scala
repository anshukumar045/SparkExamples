package project1.machinelearning
/*
A one-hot encoder maps a column of label indices to a column of binary vectors
with at most a single value. This encoding allows algorithm that except continuous features,
such as Logistic Regression, to use categorical features
 */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderEstimator, StringIndexer}
import org.apache.spark.sql.SparkSession

object OneHotEncoderApp extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("OneHotEncoderApp")
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
    .setOutputCol("categoryIndex")
    .fit(df)

  val indexed = indexer.transform(df)

  val encoder = new OneHotEncoderEstimator()
    .setInputCols(Array("categoryIndex"))
    .setOutputCols(Array("categoryVec"))
    .fit(indexed)

  val encoded = encoder.transform(indexed)
  encoded.show(false)

  val indexer1 = new StringIndexer()
    .setInputCol("name").setInputCol("address")
    .setOutputCol("categoryIndex1")
    .fit(df)

  val indexed1 = indexer1.transform(df)
  val encoder1 = new OneHotEncoderEstimator()
    .setInputCols(Array("categoryIndex1"))
    .setOutputCols(Array("categoryVec1"))
    .fit(indexed1)

  val encoded1 = encoder1.transform(indexed1)
  println("Encoder1")
  encoded1.show(false)


}
