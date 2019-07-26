package com.kanshu.sparkscala.Streaming

import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.storage.StorageLevel
import java.util.regex.Pattern
import java.util.regex.Matcher

import org.apache.log4j.{Level, Logger}

object LogparserApp extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val ssc = new StreamingContext("local[*]", "LogParser", Seconds(5))
  val lines = ssc.socketTextStream("localhost", 9182, StorageLevel.MEMORY_AND_DISK_SER)
  val requests = lines.map(e => e)
  requests.foreachRDD{
    rdd => rdd.collect().foreach(println)
  }

//  ssc.checkpoint("C:\\Users\\kanshu\\mydrive\\WMIExporter")
  ssc.start()
  ssc.awaitTermination()

}
