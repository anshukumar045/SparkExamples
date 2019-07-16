package com.kanshu.sparkscala.Rdd

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object WordCountBetter extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]","WordCountBetter")

  val input = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\book.txt")
  val words = input.flatMap(_.split("\\W+"))
  val lowercaseWords = words.map(_.toLowerCase())
  val wordCount = lowercaseWords.countByValue()
  wordCount.take(10).foreach(println)
}
