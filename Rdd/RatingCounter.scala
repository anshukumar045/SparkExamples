package com.kanshu.sparkscala.Rdd

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object RatingCounter extends App {


  Logger.getLogger("org").setLevel(Level.ERROR)

  // Create a SparkContext using every core of the local machine, named RatingsCounter
  val sc = new SparkContext("local[*]", "RatingsCounter")

  // Load up each line of the ratings data into an RDD
  val lines = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.data")

  // Convert each line to a string, split it out by tabs, and extract the third field.
  // (The file format is userID, movieID, rating, timestamp)
//  val ratings = lines.map(x => x.toString().split("\t")(2))

  val ratings = lines.map(x => x.split("\t")(2))

  lines.map(x => x.split("\t")(2)).first().foreach(println)

  // Count up how many times each value (rating) occurs
  val results = ratings.countByValue()

  // Sort the resulting map of (rating, count) tuples
  val sortedResults = results.toSeq.sortBy(_._1)

  // Print each result on its own line.
  sortedResults.foreach(println)


}
