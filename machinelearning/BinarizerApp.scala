package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.sql.SparkSession

object BinarizerApp extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().appName("BinarizerApp")
    .master("local[*]").getOrCreate()

  val data = spark.createDataFrame(Seq(
    (0, 0.1),
    (1,0.8),
    (2,0.2),
    (3,0.4),
    (4,0.9)
  )).toDF("label", "feature")

  val binarizer = new Binarizer()
    .setInputCol("feature").setOutputCol("binarized_feature").setThreshold(0.5)

  val binarizerDf = binarizer.transform(data)
  binarizerDf.show(false)

}
