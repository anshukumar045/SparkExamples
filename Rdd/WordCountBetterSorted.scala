package com.kanshu.sparkscala.Rdd

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object WordCountBetterSorted extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]","WordCountBetterSorted")

  val input = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\book.txt")
  val words = input.flatMap(x => x.split("\\W+"))
  val lowercaseWords = words.map(x => x.toLowerCase())
  val wordCounts = lowercaseWords.map(x => (x,1)).reduceByKey((x,y) => x + y)
  val wordCountSorted = wordCounts.map(x => (x._2, x._1)).sortByKey()

  for (result <- wordCountSorted) {
    val count = result._1
    val word = result._2
    println(s"$word: $count")
  }
}
