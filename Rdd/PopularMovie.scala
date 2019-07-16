package com.kanshu.sparkscala.Rdd


import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object PopularMovie extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]", "PopularMovie")

  val lines = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.data")
  val movies = lines.map(x => (x.split("\t")(1).toInt, 1))
//  movies.take(10).foreach(println)
  val movieCounts = movies.reduceByKey((x,y)=> x + y)
//  movieCounts.take(10).foreach(println)
  val flipped = movieCounts.map(x => (x._2, x._1))
  val sortedMovie = flipped.sortByKey().collect()

  sortedMovie.foreach(println)
}
