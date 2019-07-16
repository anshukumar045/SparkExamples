package com.kanshu.sparkscala.Rdd

import java.nio.charset.CodingErrorAction

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
//import scala.collection.mutable.Map


object PopularMoviesBetter extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  def loadMovieNames(): Map[Int, String] = {

    // Handle char coding issues
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

    // Create a Map of Ints
    var movieNames: Map[Int, String] = Map()

    val lines =Source.fromFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.item").getLines()
    for (line <- lines) {
      var fields = line.split('|')
      if (fields.length > 1) {
        movieNames += (fields(0).toInt -> fields(1))
      }
    }
    return movieNames
  }

  val sc = new SparkContext("local[*]","PopularMoviesBetter")

  // Create a broadcast variable of ID -> Movie name map
  val nameDict = sc.broadcast(loadMovieNames())
  val lines = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.data")
  val movies = lines.map(x => (x.split("\t")(1).toInt, 1))
  val movieCounts = movies.reduceByKey((x,y)=> x + y)
  val flipped = movieCounts.map(x => (x._2, x._1))
  val sortedMovies = flipped.sortByKey()
  val sortedMovieWithNames = sortedMovies.map(x => (nameDict.value(x._2), x._1))
  val results = sortedMovieWithNames.collect()
  results.foreach(println)
}
