package com.kanshu.sparkscala.Rdd

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object WordCount extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val sc = new SparkContext("local[*]","WordCount")
  val input = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\book.txt")

  val words = input.flatMap(x => x.split(" "))
//  words.take(10).foreach(println)
  val wordCounts = words.countByValue().take(10)

  wordCounts.foreach(println)


}
