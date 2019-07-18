package com.kanshu.sparkscala.DataframeDataSetSql

import java.nio.charset.CodingErrorAction

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._

import scala.io.{Codec, Source}
import scala.math.sqrt

object PopularMovieDataSets extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  def loadMoviesNames(): Map[Int, String] = {
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

    var movieNames: Map[Int, String] = Map()
    val lines = Source.fromFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.item").getLines()
    for (line <- lines) {
      var fields = line.split('|')
      if (fields.length > 1) {
        movieNames += (fields(0).toInt -> fields(1))
      }
    }
    return movieNames
  }

  type MovieRating = (Int, Double)
  type UserRatingPair = (Int, (MovieRating,MovieRating))

  def makePairs(userRatings: UserRatingPair): ((Int,Int),(Double,Double)) = {
    val movieRating1 = userRatings._2._1
    val movieRating2 = userRatings._2._2

    val movie1 = movieRating1._1
    val rating1 = movieRating1._2
    val movie2 = movieRating2._1
    val rating2 = movieRating2._2
    ((movie1,movie2), (rating1,rating2))
  }

  def filterDuplicates(userRating: UserRatingPair): Boolean = {
    val movieRating1 = userRating._2._1
    val movieRating2 = userRating._2._2

    val movie1 = movieRating1._1
    val movie2 = movieRating2._1

    return movie1 < movie2
  }

  type RatingPair = (Double, Double)
  type RatingPairs = Iterable[RatingPair]

  def computeCosineSimilarities(ratingPairs: RatingPairs): (Double,Int) = {
    var numPairs: Int = 0
    var sum_xx: Double = 0.0
    var sum_yy: Double = 0.0
    var sum_xy: Double = 0.0

    for (pair <- ratingPairs) {
      val ratingX = pair._1
      val ratingY = pair._2

      sum_xx += ratingX * ratingX
      sum_yy += ratingY * ratingY
      sum_xy += ratingX * ratingX
      numPairs += 1
    }

    val numerator: Double = sum_xy
    val denominator = sqrt(sum_xx) * sqrt(sum_yy)

    var score: Double = 0.0
    if (denominator != 0) {
      score = numerator / denominator
    }
    return (score, numPairs)
  }

  case class Movie(movieID: Int)

  val spark = SparkSession.builder().appName("SparkSql")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  import spark.implicits._

  val data = spark.sparkContext.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.data")
    .map(x => Movie(x.split("\t")(1).toInt))

  val moviesDS = data.toDS()
  val topMovieIDs = moviesDS.groupBy("movieID").count().orderBy($"count".desc).cache()
  topMovieIDs.show()
  val result = topMovieIDs.take(10)
  val names = loadMoviesNames()
  for (res <- result) {
    println(names(res(0).asInstanceOf[Int])+ ": "+ res(1))
  }
  spark.stop()

}
