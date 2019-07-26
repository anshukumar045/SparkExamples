package com.kanshu.sparkscala.Streaming

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.StructType
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.sql.SparkSession


object MetricsApp extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession
    .builder().master("local[*]")
    .appName("SparkStream")
    .getOrCreate()
  import spark.implicits._

  val lines = spark.readStream
    .format("socket")
    .option("host", "127.0.0.1")
    .option("port", 9182)
    .load()

  val words = lines.as[String].flatMap(_.split(" "))
  val wordCounts = words.groupBy("value").count()
  wordCounts.writeStream
    .outputMode("complete")
    .format("console")
    .start().awaitTermination(100)


}
