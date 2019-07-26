package com.kanshu.sparkscala.Streaming

import com.kanshu.sparkscala.Streaming.MetricsApp.spark
import org.apache.spark
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.StructType
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.sql.{Encoders, SparkSession}

object ExporterApp extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val userSchema = new StructType().add("name", "string").add("age", "integer")
  case class Person(name: String, age: Int)
  val people = Encoders.product[Person].schema

  val spark = SparkSession
    .builder().master("local[*]")
    .appName("SparkStream")
    .getOrCreate()
  import spark.implicits._
  val csvDf = spark
    .readStream
    .option("sep", ",")
    .schema(people)
    .csv("C:\\Users\\kanshu\\mydrive\\SparkScala\\datadirectory")

  csvDf.printSchema()
  csvDf.groupBy("name").count().writeStream.outputMode("complete").format("console").start().awaitTermination()
}
